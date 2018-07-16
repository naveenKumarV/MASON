# MASON

This project extends [VIA](http://www.robots.ox.ac.uk/~vgg/software/via/) by providing bounding box suggestions of potential objects to the annotator. The annotator can adjust the suggested bounding boxes. These suggestions ease the work of the annotator especially when the number of images to be annotated are more.

This project uses a client-server approach. The backend server used is in django. 

The suggestions come from a deep learning model. Here, we used VGG16 model pretrained on ImageNet. We used Keras with Theano backend.

To run this code, 
```python
python manage.py runserver
```
