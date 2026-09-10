"""文字意图分类服务。"""

from .classifier import IntentService, RuleIntentClassifier
from .config import IntentSettings

__all__ = ["IntentService", "IntentSettings", "RuleIntentClassifier"]
