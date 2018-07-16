from django.http import JsonResponse
from django.shortcuts import render
from .forms import ImageUploadForm
from django.apps import apps
import numpy as np
from keras.applications.vgg16 import preprocess_input
import cv2
import matplotlib.pyplot as plt
import os

def index(request):
    return render(request, 'detector/index.html')

def grab_cut_with_patch(self, patch, heat_map):
	# Grabcut mask
	# DRAW_BG = {'color': BLACK, 'val': 0}
	# DRAW_FG = {'color': WHITE, 'val': 1}
	# DRAW_PR_FG = {'color': GREEN, 'val': 3}
	# DRAW_PR_BG = {'color': RED, 'val': 2}

	self.bgdModel = np.zeros((1, 65), np.float64)
	self.fgdModel = np.zeros((1, 65), np.float64)

	mean = np.mean(heat_map[heat_map != 0])
	heat_map_high_prob = np.where((heat_map > mean), 1, 0).astype('uint8')
	heat_map_low_prob = np.where((heat_map > 0), 3, 0).astype('uint8')
	mask = heat_map_high_prob + heat_map_low_prob
	mask[mask == 4] = 1
	mask[mask == 0] = 2

	mask, bgdModel, fgdModel = cv2.grabCut(patch, mask, None, self.bgdModel, self.fgdModel, 5, cv2.GC_INIT_WITH_MASK)

	mask = np.where((mask == 2) | (mask == 0), 0, 1).astype('uint8')
	img = patch * mask[:, :, np.newaxis]
	return img



def get_bboxes(request):
    if request.method == 'POST':
        form = ImageUploadForm(request.POST, request.FILES)
        if form.is_valid():
            uploaded_image = form.save()
            model = apps.get_app_config('detector').pretrained_model
            img = cv2.imread(uploaded_image.image.path)
            os.remove(uploaded_image.image.path)
            uploaded_image.delete()
            h, w = img.shape[:2]
            img = cv2.resize(img, (224, 224))
            img = np.expand_dims(img, axis=0)
            heatmap = model.predict(img)
            heatmap = heatmap[0, :, :]
            heatmap = heatmap.sum(axis=2)
            max_val = np.max(heatmap)
            min_val = np.min(heatmap)
            heatmap = (heatmap - min_val)/(max_val - min_val)
            heatmap = heatmap[:, :]*255
            heatmap = cv2.resize(heatmap, (w, h)).astype(np.uint8)
            #thresholded_heatmap = np.zeros((h, w))
            #thresholded_heatmap[np.where(heatmap > 100)] = 1
            ret2, thresholded_heatmap = cv2.threshold(heatmap, 0, 255, 
                cv2.THRESH_BINARY+cv2.THRESH_OTSU)
            im2, contours, hierarchy = cv2.findContours(thresholded_heatmap,
                                                        cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
            regions = []
            for cnt in contours:
                x,y,w,h = cv2.boundingRect(cnt)
                regions.append({"x": x, "y": y, "h": h, "w":w})
            return JsonResponse(regions, safe=False)


def adjust_bbox(request):
    if request.method == 'POST':
        form = ImageUploadForm(request.POST, request.FILES)
        if form.is_valid():
            uploaded_image = form.save()
            model = apps.get_app_config('detector').pretrained_model
            img = cv2.imread(uploaded_image.image.path)
            os.remove(uploaded_image.image.path)
            uploaded_image.delete();
            is_cropped_img = False
            x = form.cleaned_data['x']
            y = form.cleaned_data['y']
            w = form.cleaned_data['w']
            h = form.cleaned_data['h']
            if x is not None:
            	img = img[y:y+h, x:x+w, :]
            	is_cropped_img = True
            h, w = img.shape[:2]
            img = cv2.resize(img, (224, 224))
            img = np.expand_dims(img, axis=0)
            heatmap = model.predict(img)
            heatmap = heatmap[0, :, :]
            heatmap = heatmap.sum(axis=2)
            max_val = np.max(heatmap)
            min_val = np.min(heatmap)
            heatmap = (heatmap - min_val)/(max_val - min_val)
            heatmap = heatmap[:, :]*255
            heatmap = cv2.resize(heatmap, (w, h)).astype(np.uint8)
            #thresholded_heatmap = np.zeros((h, w))
            #thresholded_heatmap[np.where(heatmap > 100)] = 1
            ret2, thresholded_heatmap = cv2.threshold(heatmap, 0, 255, 
                cv2.THRESH_BINARY+cv2.THRESH_OTSU)
            im2, contours, hierarchy = cv2.findContours(thresholded_heatmap,
                                                        cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
            largest_area = 0
            for cnt in contours:
            	area = cv2.contourArea(cnt)
            	if largest_area < area:
                	x,y,w,h = cv2.boundingRect(cnt)
                	largest_area = area
	                if is_cropped_img:
	                	x += form.cleaned_data['x']
	                	y += form.cleaned_data['y']
            region = {"x": x, "y": y, "h": h, "w":w}
            return JsonResponse(region, safe=False)       
