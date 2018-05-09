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
from define import TEXT_OUTPUT, RAW_FOLDER, TRUE_LABEL_FOLDER, DEBUG_FOLDER, IMG_INPUT
import csv
from evaluation.define import Rectangle
from img_tools import draw_rec_on_img
from evaluation.define import mergeSort
import time
import copy


class BasePipeline(IPipeline):
    __instance = None

    def __new__(cls):
        if cls.__instance is None:
            cls.__instance = super(BasePipeline, cls).__new__(cls)
            cls.__instance.__initialized = False
        return cls.__instance

    def __init__(self):
        if self.__initialized: return
        self.__initialized = True
        super().__init__()
        self.description = "A simple implementation of IPipeline"
        self.module_config_file = "config/module_config.txt"
        self.running_config_file = "config/running_config.txt"
        self.load_module_configuration()
        self.load_running_configuration()
        self.current_output_folder = ""
        self.ocr_engine = ""

    def load_module_configuration(self):
        with open(self.module_config_file, 'r') as f:
            module_details = f.readlines()

        module_details = [x.strip() for x in module_details]
        for md in module_details:
            mo_code, mo_api_url, mo_des = md.split(',')
            self.available_modules[mo_code] = {"api": mo_api_url, "des": mo_des}

    def load_running_configuration(self):
        with open(self.running_config_file, 'r') as f:
            mos_queue = f.readline()

        # Load running modules:
        mos = mos_queue.strip().split("-")
        for mo in mos:
            [mo_code, mo_param] = mo.split(":") if len(mo.split(":")) > 1 else [mo, ""]
            self.running_modules.append([mo_code, mo_param])

    def get_result(self):
        pass

    def clear_pipeline(self):
        pass

    def run_pipeline(self, img, evaluation_type=EVALTYPE.TEXT_AND_BOX.value, labels=None, img_name=IMG_INPUT):

        self.print_pipeline_info()

        # 1st: Create output folder
        out_folder = "/tmp/{}/".format(datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S'))
        mkdir(out_folder)
        self.current_output_folder = out_folder

        # 2nd: Save image input to folder to keep track
        # Also, save true label to text
        raw_out_folder = out_folder + RAW_FOLDER
        mkdir(raw_out_folder)
        cv2.imwrite(join(raw_out_folder, img_name), img)

        label_out_folder = out_folder + TRUE_LABEL_FOLDER
        mkdir(label_out_folder)

        debug_out_folder = out_folder + DEBUG_FOLDER
        mkdir(debug_out_folder)

        if labels is not None:
            labels.to_csv(path_or_buf=join(label_out_folder, TEXT_OUTPUT), sep="\t")

        # 3rd: init input data parameter
        input_data = []
        input_data.append(img)

        # 4th: load running module, set config and call run
        for mo_config in self.running_modules:
            [mo_code, mode_option] = mo_config
            
            #@TODO: hotfix here 
            if mo_code == ModuleCode.PRINT_OCR.value or mo_code == ModuleCode.HW_OCR.value:
                self.ocr_engine = mo_code
            
            module_handler = module_factory(mo_code)
            module_detail = self.available_modules[mo_code]

            mkdir(join(out_folder, mo_code))

            start_time = time.time()

            module_handler.set_module_config(mo_code, module_detail["api"], out_folder + mo_code, mode_option)
            module_handler.run_module(input_data)
            module_handler.write_output()

            input_data.clear()
            input_data = module_handler.get_output()

            print("--------- {} running in {} seconds ---------- ".format(mo_code, (time.time() - start_time)))

        # We will run the evaluation module if the following conditions are met:
        # 1. There is SEGMENTATION method
        # 2. There is OCR method
        # 3. There is true label file in correct format

        print("---------- Start Evaluating ----------- ")

        start_time = time.time()

        cer, compared_text = self.evaluate(evaluation_type)
        print("Normalized distance error is: {}".format(cer))

        print("---------- Evaluating costs : {} seconds ------------ ".format(time.time() - start_time))
        return cer

    '''
    Evaluation based on 
    '''

    def create_debug_data(self, actuals):
        # 1st: create cut debug img
        debug_img = cv2.imread(join(self.current_output_folder, RAW_FOLDER, IMG_INPUT))

        actual_color = (255, 0, 0)
        predicted_color = (0, 255, 0)
        boxes = []
        # 2nd: draw boxes
        for id, box_detail in actuals.items():
            draw_rec_on_img(debug_img, rec=box_detail['box'], text=str(id), color=actual_color)
            i = 0

            boxes[:] = []
            boxes = copy.deepcopy(box_detail['predicts'])
            mergeSort(boxes, 0, len(boxes) - 1)

            for predict_box in boxes:
                p_box = predict_box['box']
                draw_rec_on_img(debug_img, rec=p_box, text="   {}".format(str(i)), color=predicted_color)
                i += 1

        cv2.imwrite(join(self.current_output_folder, DEBUG_FOLDER, "debug.png"), debug_img)
        return True


    def evaluate(self, evaluation_type):
        evaluator_handler = evaluator_factory(evaluation_type)
        predicts, actuals = self.build_evaluator_inputs(evaluation_type)

        evaluator_handler.set_actual_values(actuals)
        evaluator_handler.set_predicted_values(predicts)
        evaluator_handler.mapping()

        cer, compared_texts = evaluator_handler.measure()
        self.create_debug_data(actuals)

        with open(join(self.current_output_folder, DEBUG_FOLDER, TEXT_OUTPUT), 'w') as out_file:
            out_file.write("{}\t{}\t{}\n".format('id', 'label', 'predict'))
            id = 0
            for (l, p) in compared_texts:
                out_file.write("{}\t{}\t{}\n".format(str(id), l, p))
                id += 1

        out_file.close()

        return cer, compared_texts



    def build_evaluator_inputs(self, option):
        predicts = {}
        actuals = {}

        if option == EVALTYPE.TEXT_AND_BOX.value:
            # 1st: build predicted input
            # 1.a Get the cut box and the corresponding ocr-text
            boxes = pd.read_csv(join(self.current_output_folder, ModuleCode.SEGMENTATION.value, TEXT_OUTPUT)
                                , sep="\t", header=0, quoting=csv.QUOTE_NONE).fillna("").to_dict("records")

            texts = pd.read_csv(join(self.current_output_folder, self.ocr_engine, TEXT_OUTPUT)
                                , sep="\t", header=0, quoting=csv.QUOTE_NONE).fillna("").to_dict("records")

            for bo, te in list(zip(boxes, texts)):
                rec_code = bo['box'][1:-1]
                x1, y1, x2, y2 = rec_code.split(",")
                box = Rectangle(int(x1), int(y1), int(x2), int(y2))
                predicts[box] = te['text']

            # 2nd: build true label input
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

                actuals[la_index] = actual_value
                la_index += 1

        return predicts, actuals

    def self_check(self):

        return True

    def print_pipeline_info(self):
        print(self.description)
        print("Selected modules: " + str(self.running_modules))
        print("Available modules: " + str(self.available_modules))


class ModuleCode(Enum):
    NORMALIZATION = "no"
    ENHANCEMENT = "en"
    BINARIZATION = "bi"
    FORM_CLASSIFICATION = "fo"
    SEGMENTATION = "se"
    PRINT_OCR = "po"
    HW_OCR = "hw"
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
        ModuleCode.PRINT_OCR.value: OcrEngine(),
        ModuleCode.HW_OCR.value: OcrEngine(),
        # ModuleCode.OCR_CORRECTION.value: OcrCorrector(),
        # ModuleCode.EVALUATION.value: Evaluator(),
    }[mo_code]


if __name__ == '__main__':
    ls = pd.read_csv("/tmp/input.tsv", sep="\t", header=0)

    b = BasePipeline()
    b.print_pipeline_info()
    img = cv2.imread("data/SG1-48-1.tif")
    b.run_pipeline(img, labels=ls, evaluation_type=EVALTYPE.TEXT_AND_BOX.value)





