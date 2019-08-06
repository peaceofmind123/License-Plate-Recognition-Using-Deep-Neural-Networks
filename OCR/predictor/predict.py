def predict_characters(net_, inputs_, use_cuda=False):
    """
    It provides the class for the image.
    :param net_: Neural Network which takes in a 3 * 50 * 50 tensor and classifies the image
    :param inputs_: tensor of size ( _, 3, 50,50)
    :param use_cuda: Boolean variable to infer the use of CUDA
    :return: character class of the image

    """

    classes = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9',
               'Ba', 'Bhe', 'Dhau', 'Ga', 'Ja', 'Ka', 'Ko', 'Lu', 'Ma', 'Me', 'Na', 'Ra', 'Sa', 'Se',
               'Cha', 'Jha', 'Yan']

    if use_cuda:
        inputs_ = inputs_.cuda()

    outputs = net_(inputs_).argmax(1).tolist()
    characters = [classes[i] for i in outputs]
    return ' '.join(characters)
