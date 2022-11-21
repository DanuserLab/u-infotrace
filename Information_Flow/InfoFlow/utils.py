
import numpy as np 

def read_video_cv2(avifile):
    
    import cv2
    import numpy as np 
    
    vidcap = cv2.VideoCapture(avifile)
    success,image = vidcap.read()
    
    vid_array = []
    
    count = 0
    success = True
    while success:
        success,image = vidcap.read()
        if success:
            vid_array.append(image)
        count += 1
        
    vid_array = np.array(vid_array)
      
    return vid_array


def mkdir(folder):
    
    import os 
    if not os.path.exists(folder):
        os.makedirs(folder)
        
    return []
    

def draw_bboxes(bboxes, ax, lw=3, color='r'):
    
    for bb in bboxes:
        x1,y1,x2,y2 = bb
        
        ax.plot([x1,x2,x2,x1,x1], 
                [y1,y1,y2,y2,y1], lw=lw, color=color)
        
    return []


def read_obj_names(textfile):
    
    classnames = []
    
    with open(textfile) as f:
        for line in f:
            line = line.strip('\n')
            if len(line)>0:
                classnames.append(line)
            
    return np.hstack(classnames)
    

def detect_imgs(infolder, ext='.tif'):
    
    import os 
    
    items = os.listdir(infolder)
    
    flist = []
    for names in items:
        if names.endswith(ext) or names.endswith(ext.upper()):
            flist.append(os.path.join(infolder, names))
    
    return np.sort(flist)


"""
Add bbox writing utilities. 
"""

def write_bboxes_voc_pred(filename, detections):
    """ format:
            cls, conf, x1, y1, x2, y2 
        where cls = string, conf = 0-1 (usually )
    """
    with open(filename, 'w') as f:
        for det in detections:
            label, score, box = det
            box = np.array(box).astype(np.int)
            
            f.write(label+'\t'+str(score)+'\t'+str(box[0])+'\t'+str(box[1])+'\t'+str(box[2])+'\t'+str(box[3])+'\n')
            
    return []


def write_bboxes_voc_gt(filename, bboxes):
    """ format:
            cls, x1, y1, x2, y2 
        where cls = string, conf = 0-1 (usually )
    """
    with open(filename, 'w') as f:
        for bbox in bboxes:
            label, box = bbox
            box = np.array(box).astype(np.int)
            
            f.write(label+'\t'+str(box[0])+'\t'+str(box[1])+'\t'+str(box[2])+'\t'+str(box[3])+'\n')
            
    return []

def read_bbox_from_file(bboxfile):
    
    f = open(bboxfile,'r')
    bboxes = []
    
    for line in f:
        line = line.strip()
        label, score, box_x, box_y, box_w, box_h = line.split()
        
        bboxes.append([label, score, int(box_x), int(box_y), int(box_w), int(box_h)])
        
    return np.array(bboxes)


def read_bboxes_from_file(bboxfiles):

    import os 
    boxes = []
    
    for f in bboxfiles:    
        box = read_bbox_from_file(f)
#        frame_no = int(((f.split('/')[-1]).split('_')[-1]).split('.txt')[0])
        frame_no = int(os.path.split(f)[-1].split('_')[2])
        boxes.append([frame_no, box])
        
    boxes = sorted(boxes, key=lambda x: x[0])
        
    return boxes



