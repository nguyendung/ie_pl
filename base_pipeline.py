from IPipeline import IPipeline
from enum import Enum
import datetime
from normalizator import Normalizator
from segmentator import Segmentator
from ocr import OcrEngine
import cv2
from os.path import join
from os import mkdir
from evaluation.evaluator_factory import evaluator_factory, EVALTYPE
import pandas as pd
from define import TEXT_OUTPUT, RAW_FOLDER, TRUE_LABEL_FOLDER, DEBUG_FOLDER, IMG_INPUT, IMG_OUTPUT, CER_FILE, VIA_FILE
import csv
from evaluation.define import Rectangle
from img_tools import draw_rec_on_img
from evaluation.define import mergeSort
import time
import copy
import configparser
from shutil import copyfile


class BasePipeline(IPipeline):
    # __instance = None

    # def __new__(cls):
    #     if cls.__instance is None:
    #         cls.__instance = super(BasePipeline, cls).__new__(cls)
    #         cls.__instance.__initialized = False
    #     return cls.__instance

    def __init__(self):
        # if self.__initialized: return
        # self.__initialized = True
        super().__init__()
        self.description = "A simple implementation of IPipeline"
        self.module_config_file = "config/module_config.txt"
        self.running_config_file = "config/running_config.txt"
        self.candidate_config_file = "config/candidate_config.txt"
        self.load_module_configuration()
        self.load_candidate_configuration()
        self.load_running_configuration()
        self.current_output_folder = ""
        self.labels = None
        self.img_name = None
        self.img_size = 0

    def load_candidate_configuration(self):
        with open(self.candidate_config_file, 'r') as f:
            cans = f.readlines()

        # Load running modules:
        for can in cans:
            des, mod_str = can.split(",")
            mos = mod_str.strip().split("-")
            a_running_config = []
            for mo in mos:
                [mo_code, mo_choice, mo_param] = mo.split(":") if len(mo.split(":")) > 1 else [mo, ""]
                a_running_config.append([mo_code, mo_choice, mo_param])
            self.cans.append([des, a_running_config])

    def get_all_running_configs(self):
        return self.cans

    def load_module_configuration(self):
        config = configparser.ConfigParser()
        config.read(self.module_config_file)

        for sec in config.sections():
            self.available_modules[sec] = {}
            for (mId, mInfo) in config.items(sec):
                api, des = mInfo.split(",")
                self.available_modules[sec][mId] = {"api": api, "des": des}

    def load_running_configuration(self):
        with open(self.running_config_file, 'r') as f:
            mos_queue = f.readline()

        # Load running modules:
        mos = mos_queue.strip().split("-")
        for mo in mos:
            [mo_code, mo_choice, mo_param] = mo.split(":") if len(mo.split(":")) > 1 else [mo, ""]
            self.running_modules.append([mo_code, mo_choice, mo_param])

    def get_result(self):
        pass

    def clear_pipeline(self):
        pass

    def run_pipeline(self, img, img_size=0, running_config=None, labels=None, img_name=IMG_INPUT):
        self.img_name = img_name
        self.img_size = img_size
        self.labels = labels
        self.running_modules = running_config
        self.print_pipeline_info()

        # 1st: Create output folder
        out_folder = "/tmp/{}_{}/".format(datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S'), img_name)
        mkdir(out_folder)
        self.current_output_folder = out_folder

        # 2nd: Save image input to folder to keep track
        # Also, save true label to text
        raw_out_folder = out_folder + RAW_FOLDER
        mkdir(raw_out_folder)
        cv2.imwrite(join(raw_out_folder, img_name), img)

        if labels is not None:
            label_out_folder = out_folder + TRUE_LABEL_FOLDER
            mkdir(label_out_folder)
            labels.to_csv(path_or_buf=join(label_out_folder, TEXT_OUTPUT), sep="\t", header=['box', 'label'])

        debug_out_folder = out_folder + DEBUG_FOLDER
        mkdir(debug_out_folder)

        # 3rd: init input data parameter
        input_data = []
        input_data.append(img)

        # 4th: load running module, set config and call run
        for mo_config in self.running_modules if running_config is None else running_config:
            [mo_code, mode_choice, mo_param] = mo_config
            
            module_handler = module_factory(mo_code)
            module_detail = self.available_modules[mo_code][mode_choice]

            mkdir(join(out_folder, mo_code))

            start_time = time.time()

            module_handler.set_module_config(mo_code, module_detail["api"], out_folder + mo_code, mo_param)

            module_handler.run_module(input_data)
            module_handler.write_output()

            input_data.clear()
            input_data = module_handler.get_output()

            print("--------- {} running in {} seconds ---------- ".format(mo_code, (time.time() - start_time)))

        cer = self.evaluate_pipeline()

        return cer, out_folder

    # We will run the evaluation module if the following conditions are met:
    # 1. There is SEGMENTATION method
    # 2. There is OCR method
    # 3. There is true label file in correct format

    def evaluate_pipeline(self, eval_type=EVALTYPE.TEXT_AND_BOX.value):

        print("---------- Start Evaluating ----------- ")
        cer = 0.0

        # if there is segmentation module AND OCR Module and Label file:
        if ModuleCode.SEGMENTATION.value in list([i[0] for i in self.running_modules]) and ModuleCode.OCR.value in list([i[0] for i in self.running_modules]):

            predicted_data = self.build_predicted_data(eval_type)

            if self.labels is not None:
                evaluator_handler = evaluator_factory(eval_type)
                labeled_data = self.build_labeled_data(eval_type)

                evaluator_handler.set_actual_values(labeled_data)
                evaluator_handler.set_predicted_values(predicted_data)
                evaluator_handler.mapping()

                cer, compared_texts = evaluator_handler.measure()

                # create debug image
                # self.create_debug_img(labeled_data)
                #
                # # create debug predict-label OCR text
                # with open(join(self.current_output_folder, DEBUG_FOLDER, TEXT_OUTPUT), 'w') as out_file:
                #     out_file.write("{}\t{}\t{}\n".format('id', 'label', 'predict'))
                #     id = 0
                #     for (l, p) in compared_texts:
                #         out_file.write("{}\t{}\t{}\n".format(str(id), l, p))
                #         id += 1
                #
                # out_file.close()
                #
                # # write measure score:
                # with open(join(self.current_output_folder, DEBUG_FOLDER, CER_FILE), 'w') as out_file:
                #     out_file.write("{}".format(str(cer)))
                #
                # create VIA csv file
                with open(join(self.current_output_folder, ModuleCode.SEGMENTATION.value, VIA_FILE), 'r') as f:
                    contents = f.readlines()
                f.close()

                contents = [c.strip() for c in contents]
                contents = [c.replace("filename", self.img_name) for c in contents]
                contents = [c.replace("imgSize", str(self.img_size)) for c in contents]

                with open(join(self.current_output_folder, DEBUG_FOLDER, VIA_FILE), 'w') as f:
                    # f.write("{}\r\n".format('#filename,file_size,file_attributes,region_count,region_id,region_shape_attributes,region_attributes'))
                    for c, t in list(zip(contents, list(predicted_data.values()))):
                        ocr_str = ',"{""OCR"":""' + "{}".format(t) + '""}"'
                        c = c[:-5]
                        c += ocr_str
                        f.write("{}\r\n".format(c))
                f.close()

                print("Normalized distance error is: {}".format(cer))
            else:
                with open(join(self.current_output_folder, ModuleCode.SEGMENTATION.value, VIA_FILE), 'r') as f:
                    contents = f.readlines()
                f.close()

                contents = [c.strip() for c in contents]
                contents = [c.replace("filename", self.img_name) for c in contents]
                contents = [c.replace("imgSize", str(self.img_size)) for c in contents]

                with open(join(self.current_output_folder, DEBUG_FOLDER, VIA_FILE), 'w') as f:
                    # f.write("{}\r\n".format('#filename,file_size,file_attributes,region_count,region_id,region_shape_attributes,region_attributes'))
                    for c, t in list(zip(contents, list(predicted_data.values()))):
                        f.write("{}\r\n".format(c))
                f.close()

                pass

                # predicted_data = self.build_predicted_data(eval_type)
                # # create VIA csv file
                # with open(join(self.current_output_folder, ModuleCode.SEGMENTATION.value, VIA_FILE), 'r') as f:
                #     contents = f.readlines()
                # f.close()
                #
                # contents = [c.strip() for c in contents]
                # contents = [c.replace("filename", self.img_name) for c in contents]
                # contents = [c.replace("imgSize", str(self.img_size)) for c in contents]
                #
                # with open(join(self.current_output_folder, DEBUG_FOLDER, VIA_FILE), 'w') as f:
                #     # f.write("{}\r\n".format('#filename,file_size,file_attributes,region_count,region_id,region_shape_attributes,region_attributes'))
                #     for c, t in list(zip(contents, list(predicted_data.values()))):
                #         ocr_str = ',"{""OCR"":""' + "{}".format(t) + '""}"'
                #         c = c[:-5]
                #         c += ocr_str
                #         f.write("{}\r\n".format(c))
                # f.close()

            ocr_file = join(self.current_output_folder, ModuleCode.OCR.value, TEXT_OUTPUT)
            dst_file = join(self.current_output_folder, DEBUG_FOLDER, TEXT_OUTPUT)
            copyfile(ocr_file, dst_file)

            self.create_linecut_img(predicted_data)

        else:
            print("Module or Label file missing")
        return cer

    def create_linecut_img(self, predicted_data):
        debug_img = cv2.imread(join(self.current_output_folder, RAW_FOLDER, self.img_name))
        predicted_color = (0, 255, 0)

        id = 1
        for rec in list(predicted_data.keys()):
            draw_rec_on_img(debug_img, rec=rec, text=str(id), color=predicted_color)
            id += 1
        cv2.imwrite(join(self.current_output_folder, DEBUG_FOLDER, IMG_OUTPUT), debug_img)

    def create_debug_img(self, labeled_data):
        # 1st: create cut debug img
        debug_img = cv2.imread(join(self.current_output_folder, RAW_FOLDER, self.img_name))

        actual_color = (255, 0, 0)
        predicted_color = (0, 255, 0)
        boxes = []

        # 2nd: draw boxes
        for id, box_detail in labeled_data.items():
            draw_rec_on_img(debug_img, rec=box_detail['box'], text=str(id), color=actual_color)
            i = 0

            boxes[:] = []
            boxes = copy.deepcopy(box_detail['predicts'])
            mergeSort(boxes, 0, len(boxes) - 1)

            for predict_box in boxes:
                p_box = predict_box['box']
                draw_rec_on_img(debug_img, rec=p_box, text="   {}".format(str(i)), color=predicted_color)
                i += 1

        cv2.imwrite(join(self.current_output_folder, DEBUG_FOLDER, IMG_OUTPUT), debug_img)
        return True



    def build_predicted_data(self, option):
        predicts = {}

        if option == EVALTYPE.TEXT_AND_BOX.value:
            boxes = pd.read_csv(join(self.current_output_folder, ModuleCode.SEGMENTATION.value, TEXT_OUTPUT)
                                , sep="\t", header=0, quoting=csv.QUOTE_NONE).fillna("").to_dict("records")

            texts = pd.read_csv(join(self.current_output_folder, ModuleCode.OCR.value, TEXT_OUTPUT)
                                , sep="\t", header=0, quoting=csv.QUOTE_NONE).fillna("").to_dict("records")

            for bo, te in list(zip(boxes, texts)):
                p1, p2, p3, p4 = bo['box'].split(",")
                x1, y1 = p1[1:-1].split()
                x2, y2 = p3[1:-1].split()
                box = Rectangle(int(float(x1)), int(float(y1)), int(float(x2)), int(float(y2)))
                predicts[box] = te['text']

        return predicts

    def build_labeled_data(self, option):
        labeled_data = {}
        if option == EVALTYPE.TEXT_AND_BOX.value:
            labels = pd.read_csv(join(self.current_output_folder, TRUE_LABEL_FOLDER, TEXT_OUTPUT)
                                 , sep="\t", header=0, quoting=csv.QUOTE_NONE).fillna("").to_dict("records")

            la_index = 0
            for la in labels:
                rec_code = la['box'][1:-1]
                x1, y1, x2, y2 = rec_code.split(",")
                box = Rectangle(int(x1), int(y1), int(x2), int(y2))

                actual_value = {}
                actual_value['box'] = box
                actual_value['label'] = la['label']
                actual_value['predicts'] = []

                labeled_data[la_index] = actual_value
                la_index += 1

        return labeled_data


    def print_pipeline_info(self):
        # print(self.description)
        # print("Available modules: " + str(self.available_modules))
        print("Running modules: " + str(self.running_modules))

    def self_check(self):

        return True


class ModuleCode(Enum):
    NORMALIZATION = "no"
    ENHANCEMENT = "en"
    BINARIZATION = "bi"
    FORM_CLASSIFICATION = "fo"
    SEGMENTATION = "se"
    OCR = "ocr"
    OCR_CORRECTION = "oc"
    EVALUATION = "ev"
    TRUE_LABEL = 'la'
    
def module_factory(mo_code):
    return {
        ModuleCode.NORMALIZATION.value: Normalizator(),
        # ModuleCode.ENHANCEMENT.value: Enhancer(),
        # ModuleCode.BINARIZATION.value: Binarizator(),
        # ModuleCode.FORM_CLASSIFICATION.value: FormClassifier(),
        ModuleCode.SEGMENTATION.value: Segmentator(),
        ModuleCode.OCR.value: OcrEngine(),
        # ModuleCode.OCR_CORRECTION.value: OcrCorrector(),
        # ModuleCode.EVALUATION.value: Evaluator(),
    }[mo_code]


if __name__ == '__main__':
    ls = pd.read_csv("/tmp/input.tsv", sep="\t", header=0)

    b = BasePipeline()
    b.print_pipeline_info()
    img = cv2.imread("data/SG1-48-1.tif")
    b.run_pipeline(img, labels=ls, evaluation_type=EVALTYPE.TEXT_AND_BOX.value)





