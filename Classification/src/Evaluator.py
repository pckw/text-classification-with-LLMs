from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score


class Evaluator():
    def __init__(self) -> None:
        pass

    def evaluate(self, test, prediction):
        self.x = [i for i,j in zip(test['data'], test[prediction]) if j >= 0]
        self.y = [i for i,j in zip(test['target'], test[prediction]) if j >= 0]
        self.topic_names = test['target_names']
        self.y_pred = [i for i in test[prediction] if i >= 0]
        print(f"Number of records: {len(test['data'])}")
        if test[prediction].count(-1) > 0:
            print(f"Number of unclassified records: {test[prediction].count(-1)}")
        if test[prediction].count(-2) > 0:
            print(f"Number of records filtered by Azures content filter: {test[prediction].count(-2)}")

        print(classification_report(
            self.y,
            self.y_pred,
            labels=range(len(self.topic_names)),
            target_names=self.topic_names)
        )
        # compute recall, precision, f1-score
        self.accuracy = accuracy_score(self.y, self.y_pred)
        self.precision = precision_score(self.y, self.y_pred, average='macro')
        self.recall = recall_score(self.y, self.y_pred, average='macro')
        self.f1 = f1_score(self.y, self.y_pred, average='macro')