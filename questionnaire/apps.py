from django.apps import AppConfig
from django.conf import settings
import pandas as pd
import logging
import os

logger = logging.getLogger(__name__)


class QuestionnaireConfig(AppConfig):
    name = 'questionnaire'

    def ready(self):
        # Initialize IPIP 50 questionnaire on import
        path = os.path.join(settings.BASE_DIR, 'data', 'IPIP50.csv')
        questions = pd.read_csv(path)

        try:
            for (q, trait, pos_key) in questions.iloc:
                IPIPQuestion = self.get_model('IPIPQuestion')
                question = IPIPQuestion(question_text=q, scored_trait=int(trait), positive_keyed=bool(pos_key))
                question.save()
        except:
            logger.info('Tried to initialize IPIP questions, but they already exist in the database.')


