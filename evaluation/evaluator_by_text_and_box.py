from evaluation.IEvaluator import IEvaluator
from evaluation.define import area, mergeSort
from nltk.metrics.distance import edit_distance
import copy

'''
Format for:
- actuals will be key:value in which key is id; value is {"box":Rectangle;"label":text,"predicts":[]}
- predicts will be key:value in which key is Rectangle; value is text 
'''


class EvaluatorByTextAndBox(IEvaluator):
    def measure(self):
        error_distance = 0
        boxes = []
        character_count = 0.0
        compared_texts = []
        for ak, av in self.actuals.items():
            character_count += len(av['label'])
            boxes[:] = []
            boxes = copy.deepcopy(av['predicts'])

            mergeSort(boxes, 0, len(boxes) - 1)

            ss = []
            for x in boxes:
                ss.append(str(x['text']))
            merged_predicted_text = ''.join(ss)
            box_error_distance = edit_distance(av['label'].replace(" ", ""), merged_predicted_text.replace(" ", ""))
            compared_texts.append((av['label'].replace(" ", ""), merged_predicted_text.replace(" ", "")))
            error_distance += box_error_distance
        normalized_error_distance = error_distance/character_count
        return normalized_error_distance, compared_texts

    def mapping(self):
        for pk, pv in self.predicteds.items():
            cut_box = pk
            matched_box_id = -1
            max_overlapped_area = 0
            for ak, av in self.actuals.items():
                # print(ak, av)
                label_box = av["box"]
                overlapped_area = area(cut_box, label_box)
                if overlapped_area > max_overlapped_area:
                    max_overlapped_area = overlapped_area
                    matched_box_id = ak
            if matched_box_id >= 0:
                # print("Cut box {} is matched with label box {}".format(cut_box, self.actuals[matched_box_id]['box']))
                self.actuals[matched_box_id]["predicts"].append({"box": pk, "text": pv})
            else:
                pass
                # print("Cut box {} does not mated any label box".format(cut_box))




