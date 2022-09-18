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

datasrc = 'dedun.hi.no:/data/deep/CoastVision/Updated_dataset_object_detection_and_segmentation'

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
                    pts = [ (e.attrib['xtl'], e.attrib['ytl']) ,
                            (e.attrib['xbr'], e.attrib['ybr']) ]
                else:
                    pass
                sex = None
                species = None
                for attr in e:
                    assert(attr.tag == 'attribute')
                    if attr.attrib['name'] == 'species':
                        species = attr.text
                    if attr.attrib['name'] == 'sex' and not attr.text == 'unknown':
                        sex = "_"+attr.text
                if species == "cuckoo" or species == "corkwing" and sex is not None: species += sex
                cur.append((imfile, species , pts))
            res[mydir] += cur
    # print(res)
    return res

def rename(fn):
    return fn.replace(' ','-')

class Data:
    def __init__(self, conf, mypath):
        self.config = conf
        self.mypath = mypath

    def get(self):
        '''Download and unpack the data'''

        if os.path.exists('images'):
            print('The "images" directory exists already - skipping data download!')
            return

        if os.path.exists(os.path.basename(datasrc)):
            print('Data already downloaded - skipping')
        else:
            print(f'Syncing data from {datasrc}')
            os.system(f'rsync -a {datasrc} .')

        # os.system(f'unzip -u {os.path.basename(datasrc)}')
        prefixdir = os.path.basename(datasrc)

        # move:
        #   mv Nest20_04\&0506 Nest20_04-0506
        #   mv Nest20_0306     Nest_030620
        #   mv Nest18_BOUNDINGBOXES 'Nest18 BOUNDINGBOXES'

        annos = extract_xml([prefixdir+'/'+s for s in self.config['annotations']])
        os.mkdir("images")
        with open('annotations.csv', 'w') as f:
            for mydir in annos:
                print(mydir, len(annos[mydir]))
                # todo: copy images, create masks, output annotations to f
                # for im,spec,pts in anns:
                for im, spec, pts in annos[mydir]:
                    imname = rename(im)
                    xs = [ float(x) for (x,y) in pts ]
                    ys = [ float(y) for (x,y) in pts ]
                    bbox = ( min(xs), min(ys), max(xs), max(ys) )
                    if os.path.exists(f'{prefixdir}/{mydir}/{im}'):
                        os.system(f'cp "{prefixdir}/{mydir}/{im}" images/{imname}')
                        f.write(f'{imname}\t{spec}\t{bbox}\n')
                    else:
                        print(f"Can't find: {prefixdir}/{mydir}/{im}!")

    def validate(self):
        pass

if __name__ == '__main__':
    d = Data(defaultconf, '')
    d.get()
    d.validate()
