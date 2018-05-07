from enum import Enum
from evaluation.evaluator_by_text_and_box import EvaluatorByTextAndBox
from evaluation.evaluator_by_text_only import EvaluatorByTextOnly


class EVALTYPE(Enum):
    TEXT_ONLY = 1
    TEXT_AND_BOX = 2


def evaluator_factory(mode):
    return {
        EVALTYPE.TEXT_AND_BOX.value: EvaluatorByTextAndBox(),
        EVALTYPE.TEXT_ONLY.value: EvaluatorByTextOnly()
    }[mode]

