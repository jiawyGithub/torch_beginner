
from matplotlib.pyplot import text


def get_fashion_mnist_pred_labels(pred,labels):
    text_labels = ['t-shirt', 'trouser', 'pullover', 'dress', 'coat',
                   'sandal', 'shirt', 'sneaker', 'bag', 'ankle boot']
    text = []
    for i in range(labels.size(0)):
        text.append(f"{text_labels[pred[i]]}/{text_labels[labels[i]]}")
    return text

def get_fashion_mnist_labels(labels):
    text_labels = ['t-shirt','trouser','pullover','dress','coat','sandal',
                  'shirt','sneaker','bag','ankle boost'
                  ]
    return [text_labels[int(i)] for i in labels]
