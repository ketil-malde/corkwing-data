# import numpy as np
# import matplotlib.pyplot as plt
# import cv2
import os
import os.path
import xml.etree.ElementTree as et
import cv2
import numpy as np

defaultconf = {
    'annotations': ['All_annotations']  # ['Nest1_finished', '16_Nest2']
    }

datasrc = 'dedun:/data/deep/CoastVision/Updated_dataset_object_detection_and_segmentation.zip'

def collectpts(pts):
    '''Convert to list of pairs of floats'''
    return [(float(x), float(y)) for x,y in [s.split(',') for s in pts.split(';')]]

def extract_xml(anns):
    '''Parse xml annotation, output csv table and masks'''
    res = {}
    assert(isinstance(anns,list))

    for fn in anns:
        print(f'Parsing  {fn}.xml')
        tree = et.parse(f'{fn}.xml')
        r = tree.getroot()
        tasks = {}
        for ts in r.findall("meta/project/tasks/task"):
            mydir = ts.find("name").text
            tasks[ts.find("id").text] = mydir
            res[mydir] = []

        for ch in r.findall("image"):
            cur = []
            imfile = ch.attrib['name']
            mydir = tasks[ch.attrib['task_id']]
            for e in ch:
                if e.tag == 'polygon':
                    # image/polygon/attribute(sex,body_shape_guessed,main_orientation,completely_in_frame,type)
                    pts = collectpts(e.attrib['points'])
                elif e.tag == 'box':
                    pts = [(e.attrib['xtl'],e.attrib['ytl']), [(e.attrib['xbr'],e.attrib['ybr']) ] ]
                else:
                    pass
                sex = ''
                for attr in e:
                    assert(attr.tag == 'attribute')
                    if attr.attrib['name'] == 'species':
                        species = attr.text
                    if attr.attrib['name'] == 'sex' and not attr.text == 'unknown':
                        sex = "_"+attr.text
                cur.append((imfile, species+sex, pts))
            res[mydir].append(cur)
    print(res)
    return res

def rename(fn):
    return fn.replace(' ','-')

class Data:
    def __init__(self, conf, mypath):
        self.config = conf
        self.mypath = mypath

    def get(self):
        '''Download and upack the data'''
        if os.path.exists('images'):
            print('The "images" directory exists already - skipping data download!')
            return

        if os.path.exists(os.path.basename(datasrc)):
            print('Data already downloaded - skipping')
        else:
            os.system(f'scp {datasrc} .')
        os.system(f'unzip -u {os.path.basename(datasrc)}')

        os.mkdir("images")
        os.mkdir("instance_masks")
        i=0
        with open('annotations.csv', 'w') as f:
            for mydir, anns in extract_xml(self.config['annotations']):
                print(mydir, len(anns))
                # todo: copy images, create masks, output annotations to f
                for im,spec,pts in anns:
                    imname = rename(im)
                    i+=1
                    maskname=f'{imname}_{spec}_{i}.png'
                    xs = [ x for (x,y) in pts ]
                    ys = [ y for (x,y) in pts ]
                    bbox = ( min(xs), min(ys), max(xs), max(ys) )
                    f.write(f'{imname}\t{spec}\t{bbox}\t{maskname}\n')
                    os.system(f'cp "Datasetforsegmentation/{mydir}/{im}" images/{imname}')
                    tmpimg = cv2.imread(f'images/{imname}')
                    H,W,C = tmpimg.shape
                    mask = np.zeros((H,W))
                    p = np.array(pts, dtype=np.int32)
                    cv2.fillPoly(mask, [p], 1)
                    cv2.imwrite(f'instance_masks/{maskname}', mask*255)
                    
    def validate(self):
        pass

if __name__ == '__main__':
    d = Data(defaultconf, '')
    d.get()
    d.validate()
