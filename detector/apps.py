from django.apps import AppConfig
import sys
import os
import detector



class DetectorConfig(AppConfig):
    name = 'detector'
    pretrained_model = None
    graph = None

    def ready(self):
    	if 'runserver' not in sys.argv:
    		return True

    	from keras.models import load_model
    	if DetectorConfig.pretrained_model is None:
	    	DetectorConfig.pretrained_model = load_model(os.path.dirname(detector.__file__) + 
	    		'/pretrained_models/vgg16.h5')
