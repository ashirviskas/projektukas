import numpy as np
import xml.etree.ElementTree as ET
import struct
import matplotlib.pyplot as plt
import os
import h5py

class Bandinys:
    def __init__(self, header_root = None, data_filepath = None):
        self.header_root = header_root
        self.data_filepath = data_filepath
        # Generator parameters
        self.frequency = None
        self.number_of_periods = None
        self.excitation_voltage = None
        # Scanner parameters
        self.s_start = []
        self.s_end = []
        self.s_step = []
        self.s_order = []
        self.s_use = []
        # ADC parameters
        self.adc_start_time = None
        self.adc_stop_time = None
        self.sampling_frequency = None
        self.channels = []
        # Amplifier parameters
        self.static_gain = []
        self.dynamic_gain = []
        self.k_gain = []
        self.b_scan = None
        self.construct_metadata()
        self.construct_data()

    def construct_metadata(self):
        if not self.header_root:
            return False

        generator_parameters = self.header_root.find("GeneratorParameters")
        self.frequency = int(generator_parameters.find('Frequency').text)
        self.number_of_periods = int(generator_parameters.find('NumberOfPeriods').text)
        self.excitation_voltage = int(generator_parameters.find('ExcitationVoltage').text)

        for child in self.header_root.find("ScannerParameters"):
            if child.tag[:4] == "Axis":
                self.s_start.append(int(child.get("start")))
                self.s_end.append(int(child.get("end")))
                self.s_step.append(float(child.get("step")))
                self.s_order.append(int(child.get("order")))
                self.s_use.append((True if child.get("use") == "True" else False))
            else:
                break

        adc_parameters = self.header_root.find("ADCParameters")
        self.adc_start_time = int(adc_parameters.find("ADCstartTime").text)
        self.adc_stop_time = int(adc_parameters.find("ADCstopTime").text)
        self.sampling_frequency = int(adc_parameters.find("SamplingFrequency").text)
        for child in adc_parameters:
            if child.tag[:4] == "Chan":
                self.channels.append((True if child.text == "True" else False))

        amplifier_parameters = self.header_root.find("AmplifierParameters")
        self.static_gain.append(int(amplifier_parameters.find("Ch1StaticGain").text))
        self.static_gain.append(int(amplifier_parameters.find("Ch2StaticGain").text))

        self.dynamic_gain.append(int(amplifier_parameters.find("Ch1DynamicGain").text))
        self.dynamic_gain.append(int(amplifier_parameters.find("Ch2DynamicGain").text))

        for i in range(2):
            self.k_gain.append(10**((self.static_gain[i] + self.dynamic_gain[i] + 7 + 13.4)/20))

    def construct_data(self):
        if not self.data_filepath:
            return False
        y1 = []
        y2 = []
        if max(self.s_order) == 0:
            n_samples = (self.adc_stop_time - self.adc_start_time) * self.sampling_frequency
            f = open(self.data_filepath, 'rb')
            s_position = []
            for i in range(8):
                s_position.append(struct.unpack("d", f.read(8))[0]) # Read 8 bytes (64 bits) from file and unpack into double
            s_start_time = struct.unpack("d", f.read(8))[0]
            s_stop_time = struct.unpack("d", f.read(8))[0]
            n_samples2 = struct.unpack("i", f.read(4))[0] # 32 bits integer
            for i in range(n_samples):
                num = struct.unpack("h", f.read(2))[0] # Unpack 16 bit integer
                y1.append(num)
            y1 = (np.array(y1).transpose() - 512)/521*0.5/self.k_gain[0]

            if (self.channels[0] and self.channels[1]):
                s_position = []
                for i in range(8):
                    s_position.append(
                        struct.unpack("d", f.read(8))[0])  # Read 8 bytes (64 bits) from file and unpack into double
                s_start_time = struct.unpack("d", f.read(8))[0]
                s_stop_time = struct.unpack("d", f.read(8))[0]
                n_samples2 = struct.unpack("i", f.read(4))[0]  # 32 bits integer
                for i in range(n_samples):
                    num = struct.unpack("h", f.read(2))[0]  # Unpack 16 bit integer
                    y2.append(num)
                y2 = (np.array(y2).transpose() - 512) / 521 * 0.5 / self.k_gain[1]
                # print(n_samples2)
            f.close()

        elif max(self.s_order) == 1:
            k_axis = 0
            b_scan = []
            b_scan2 = []
            for k_axis in range(6):
                if (self.s_order[k_axis] != 1):
                    continue
                else:
                    break
            n_steps = int((self.s_end[k_axis] - self.s_start[k_axis])/self.s_step[k_axis] + 1)
            n_samples = (self.adc_stop_time - self.adc_start_time) * self.sampling_frequency
            f = open(self.data_filepath, 'rb')
            for ks in range(n_steps):
                s_position = []
                for i in range(8):
                    s_position.append(
                        struct.unpack("d", f.read(8))[0])  # Read 8 bytes (64 bits) from file and unpack into double
                s_start_time = struct.unpack("d", f.read(8))[0]
                s_stop_time = struct.unpack("d", f.read(8))[0]
                n_samples2 = struct.unpack("i", f.read(4))[0]  # 32 bits integer
                y1 = []
                for i in range(n_samples):
                    num = struct.unpack("h", f.read(2))[0]  # Unpack 16 bit integer
                    y1.append(num)
                y1 = (np.array(y1).transpose() - 512) / 521 * 0.5 / self.k_gain[0]
                b_scan.append(y1)
                if (self.channels[0] and self.channels[1]):
                    s_position = []
                    for i in range(8):
                        s_position.append(
                            struct.unpack("d", f.read(8))[0])  # Read 8 bytes (64 bits) from file and unpack into double
                    s_start_time = struct.unpack("d", f.read(8))[0]
                    s_stop_time = struct.unpack("d", f.read(8))[0]
                    n_samples2 = struct.unpack("i", f.read(4))[0]  # 32 bits integer
                    y2 = []
                    for i in range(n_samples):
                        num = struct.unpack("h", f.read(2))[0]  # Unpack 16 bit integer
                        y2.append(num)
                    y2 = (np.array(y2).transpose() - 512) / 521 * 0.5 / self.k_gain[1]
                    b_scan2.append(y2)
            f.close()
            b_scan = np.array(b_scan)
            self.b_scan = b_scan
            # plt.pcolormesh(b_scan)
            # plt.show()

    def display_data(self):
        if self.b_scan is not None:
            plt.pcolormesh(self.b_scan)
            plt.show()
        else:
            return False


def get_all_files_in_dir(directory):
    filenames = []
    filepath = './advice2/' + directory
    samples_dir = os.listdir(filepath)
    for filename in samples_dir:
        if os.path.isfile(filepath + '/' + filename) and filename[-4:] == ".uld":
            filenames.append(filepath + "/" + filename[:-4])
    return filenames


def read_headerfile(filepath):
    tree = ET.parse(filepath)
    root = tree.getroot()
    # for child in root:
    #     print(child.tag, child.attrib)
    return root


def files_to_hdf5(data_filename = "data_0.hdf5"):
    data_file = h5py.File(data_filename, 'w')
    bandiniai = []
    bandiniai.append(["bandinys1", ["3g", "5g"]])
    bandiniai.append(["bandinys2", ["3g", "5g"]])
    bandiniai.append(["bandinys3", ["3g", "5g"]])
    filepaths = []
    for b in bandiniai:
        filepaths.append(get_all_files_in_dir(b[0] + "/" + b[1][0]))
        filepaths.append(get_all_files_in_dir(b[0] + "/" + b[1][1]))
    print(filepaths)

    for i in range(0, len(filepaths)):
        for j in range(0, len(filepaths[i])):
            print(filepaths[i][j])
            header_root = read_headerfile(filepaths[i][j] + ".ulh")
            bandinys = Bandinys(header_root, filepaths[i][j] + ".uld")
            dataset_one = data_file.create_dataset(filepaths[i][j][1:], data=bandinys.b_scan)
            print(dataset_one.name)
    data_file.close()

if __name__ == "__main__":
    files_to_hdf5()

