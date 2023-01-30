import numpy as np

def gen_bx(img_dim,bx,dilate=False,rand_dilate=False,**kwargs):
    
    # img_dim: image dimension
    # bx: txt record bx same as yolov5
    # dilate=True then need a input key "dilate_factor"
    # rand_dilate=True then need a input key "max_dilate_factor"
    
    #img_dim = np.array((511,511))
    img_dim2 = np.array(img_dim)-1
    x, y, w, h = img_dim2[1]*(bx[1]-bx[3]/2), img_dim2[0]*(bx[2]-bx[4]/2), img_dim2[1]*bx[3], img_dim2[0]*bx[4]
    x, y, w, h = int(round(x))+1,  int(round(y))+1, int(round(w)), int(round(h))
    X, Y = [x, x+w-1], [y, y+h-1]
    
    
    if dilate==True or rand_dilate==True:
        if dilate==True:
            dia = kwargs['dilate_factor']                       
        elif rand_dilate==True:
            dia = np.random.randint(0,kwargs['max_dilate_factor']+1)
        # make bx dilation    
        X[0],X[1] = X[0]-dia,X[1]+dia
        Y[0],Y[1] = Y[0]-dia,Y[1]+dia 
        if X[0]<1: X[0]=1
        if X[1]>img_dim[1]: X[1]=img_dim[1]
        if Y[0]<1: Y[0]=1
        if Y[1]>img_dim[0]: Y[1]=img_dim[0]
    
    # mask
    mask = np.zeros(img_dim)
    mask[Y[0]-1:Y[1],X[0]-1:X[1]] = 1
    
    return mask

def extract_bboxes(mask):

    """Compute bounding boxes from masks.
    mask: [height, width, num_instances]. Mask pixels are either 1 or 0.
    Returns: bbox array [num_instances, (y1, x1, y2, x2)].
    """

    boxes = np.zeros([mask.shape[-1], 4], dtype=np.int32)
    boxes_yolo = np.zeros([mask.shape[-1], 5])
    boxes_yolo2 = np.zeros([mask.shape[-1], 5])

    for i in range(mask.shape[-1]):
        m = mask[:, :, i]

        # Bounding box.
        horizontal_indicies = np.where(np.any(m, axis=0))[0]
        #print("np.any(m, axis=0)",np.any(m, axis=0))
        #print("p.where(np.any(m, axis=0))",np.where(np.any(m, axis=0)))
        vertical_indicies = np.where(np.any(m, axis=1))[0]

        if horizontal_indicies.shape[0]:
            x1, x2 = horizontal_indicies[[0, -1]]
            y1, y2 = vertical_indicies[[0, -1]]
            # x2 and y2 should not be part of the box. Increment by 1.
            x2 += 1
            y2 += 1
        else:
            # No mask for this instance. Might happen due to
            # resizing or cropping. Set bbox to zeros
            x1, x2, y1, y2 = 0, 0, 0, 0

        boxes[i] = np.array([y1, x1, y2, x2])
        
        boxes_yolo[i] = np.array([i,
                                  (x1+(x2+1-x1)/2)/mask.shape[1],
                                  (y1+(y2+1-y1)/2)/mask.shape[0],
                                  ((x2-x1))/mask.shape[1],
                                  ((y2-y1))/mask.shape[0]])
        
        boxes_yolo2[i] = np.array([i,
                                  (x1+(x2+1-x1)/2),
                                  (y1+(y2+1-y1)/2),
                                  ((x2-x1)),
                                  ((y2-y1))])

    return boxes.astype(np.int32), boxes_yolo, boxes_yolo2