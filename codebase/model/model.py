import torch.nn as nn

import os, sys, inspect
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
sys.path.insert(0,currentdir)

from feature_extraction import VGG_FeatureExtractor,ResNet_FeatureExtractor
from sequence_modeling import BidirectionalLSTM
from utils import CTCLabelConverter

class Model(nn.Module):

    def __init__(self, opt):
        super(Model, self).__init__()
        self.opt = opt
        if self.opt.rgb:
            self.opt.input_channel = 3

        """ FeatureExtraction """
        if opt.feature_extractor == 'VGG':
            self.FeatureExtraction = VGG_FeatureExtractor(self.opt.input_channel, self.opt.output_channel)
        else:
            self.FeatureExtraction = ResNet_FeatureExtractor(self.opt.input_channel, self.opt.output_channel)

        self.FeatureExtraction_output = self.opt.output_channel  # int(imgH/16-1) * 512
        self.AdaptiveAvgPool = nn.AdaptiveAvgPool2d((None, 1))  # Transform final (imgH/16-1) -> 1

        """ Sequence modeling"""
        
        self.SequenceModeling = nn.Sequential(
                BidirectionalLSTM(self.FeatureExtraction_output, self.opt.hidden_size, self.opt.hidden_size),
                BidirectionalLSTM(self.opt.hidden_size, self.opt.hidden_size, self.opt.hidden_size))
        self.SequenceModeling_output = self.opt.hidden_size
       
        """ model configuration """
        converter = CTCLabelConverter(self.opt.character)
        opt.num_class = len(converter.character)

        """ Prediction """
        self.Prediction = nn.Linear(self.SequenceModeling_output, opt.num_class)
        

    def forward(self,x):
        input, text = x[0], x[1]

        """ Feature extraction stage """
        visual_feature = self.FeatureExtraction(input)
        visual_feature = self.AdaptiveAvgPool(visual_feature.permute(0, 3, 1, 2))  # [b, c, h, w] -> [b, w, c, h]
        visual_feature = visual_feature.squeeze(3)

        """ Sequence modeling stage """
        contextual_feature = self.SequenceModeling(visual_feature)

        """ Prediction stage """
        prediction = self.Prediction(contextual_feature.contiguous())

        return prediction
