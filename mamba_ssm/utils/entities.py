import os
from typing import List, Tuple

class Page:
    name = "Page"
    __version__ = "v1.0"

    def __init__(self, image, page_number):
        self.image = image
        self.page_number = page_number
        self.doctype = 'other'
        self.tables = []
        self.lines = []
        self.words = []
        self.fields = []
        self.keys = []
        self.sdmgr_lines = []
        self.kie_lines = []
        self.multi_values = {}
        self.single_values = {}

    @property
    def width(self):
        return self.image.shape[1]

    @property
    def height(self):
        return self.image.shape[0]


class Document:
    name = "Document"
    __version__ = "v1.0"

    def __init__(self):
        self.path = ""
        self.content_type = None
        self.pages = []

    def add_page(self, page: Page):
        self.pages.append(page)

    def __call__(self):
        return {
            "path": self.path,
            "content_type": self.content_type,
            "pages": [obj() for obj in self.pages],
        }


class Block:
    name = "Block"
    __version__ = "v1.0"

    def __init__(self, xmin, ymin, xmax, ymax):
        self.xmin = float("%.3f" % xmin)  # normalize [0, 1]
        self.ymin = float("%.3f" % ymin)  # normalize [0, 1]
        self.xmax = float("%.3f" % xmax)  # normalize [0, 1]
        self.ymax = float("%.3f" % ymax)  # normalize [0, 1]s

    def __call__(self):
        return {
            "xmin": self.xmin,
            "xmax": self.xmax,
            "ymin": self.ymin,
            "ymax": self.ymax,
            "type": self.name,
        }

    @property
    def x_center(self):
        return (self.xmin + self.xmax) / 2

    @property
    def y_center(self):
        return (self.ymin + self.ymax) / 2

    @property
    def width(self):
        return self.xmax - self.xmin

    @property
    def height(self):
        return self.ymax - self.ymin

    @property
    def coordinate(self):
        return (self.xmin, self.ymin, self.xmax, self.ymax)


class TextBox(Block):
    name = "TextBox"
    __version__ = "v1.0"

    def __init__(self, block, text, conf, id=None, language=""):
        super().__init__(block.xmin, block.ymin, block.xmax, block.ymax)
        self.id = id
        self.characters = []
        self.text = text
        self.conf = conf
        self.raw_text = ""
        self.class_name = None
        self.class_score = None
        self.language = language
        self.metadata = {}

    def update_class(self, id, conf):
        self.class_name = id
        self.class_score = conf

    def add_char(self, other):
        self.xmin = min(self.xmin, other.xmin)
        self.ymin = min(self.ymin, other.ymin)
        self.xmax = max(self.xmax, other.xmax)
        self.ymax = max(self.ymax, other.ymax)
        self.characters.append(other)
        self.characters.sort(key=lambda x: x.xmin)
        self.text = "".join([x.text for x in self.characters]).strip()

    def __call__(self):
        json_result = super().__call__()
        json_result["text"] = self.text
        json_result["confidence"] = self.conf
        if self.class_name is not None:
            json_result["classId"] = self.class_name
        if self.class_score is not None:
            json_result["classConfidence"] = self.class_score

        return json_result


class Line:
    def __init__(self, init_box, id=None):
        self.id = id
        self.textboxes = [init_box]
        self.class_name = None
        self.class_score = 1.0

    def add_textbox(self, box):
        self.textboxes.append(box)
        self.textboxes.sort(key=lambda x: x.xmin)

    @property
    def text(self):
        return " ".join([x.text for x in self.textboxes]).strip()

    @property
    def conf(self):
        return sum([x.conf for x in self.textboxes]) / len(self.textboxes)

    @property
    def xmin(self):
        return min(o.xmin for o in self.textboxes)

    @property
    def ymin(self):
        return min(o.ymin for o in self.textboxes)

    @property
    def xmax(self):
        return max(o.xmax for o in self.textboxes)

    @property
    def ymax(self):
        return max(o.ymax for o in self.textboxes)

    @property
    def width(self):
        return self.xmax - self.xmin

    @property
    def height(self):
        return self.ymax - self.ymin

    @property
    def x_center(self):
        return (self.xmax + self.xmin) / 2

    @property
    def y_center(self):
        return (self.ymax + self.ymin) / 2

class Key(Line):
    def __init__(self, word_init, id=None):
        self.id = id
        self.textboxes = [word_init]
        self.values = []
        self.text_postprocess = ""
        self.class_name = None
        self.class_score = 1.0

    def add_value(self, value):
        self.values.append(value)

class Value(Line):
    def __init__(self, word_init, id=None):
        self.id = id
        self.textboxes = [word_init]
        self.text_postprocess = ""
        self.class_name = None
        self.class_score = 1.0

class Field:
    def __init__(self, init_box):
        self.textboxes = [init_box]

    def add_textbox(self, box):
        self.textboxes.append(box)
        self.textboxes.sort(key=lambda x: (x.ymin, x.xmin))

    def add_other(self, other):
        self.textboxes += other.textboxes
        self.textboxes.sort(key=lambda x: (x.ymin, x.xmin))

    @property
    def class_score(self):
        return sum([x.class_score for x in self.textboxes]) / len(self.textboxes)

    @property
    def class_name(self):
        return self.textboxes[0].class_name

    @property
    def text(self):
        return " ".join([x.text for x in self.textboxes]).strip()

    @property
    def xmin(self):
        return min(o.xmin for o in self.textboxes)

    @property
    def ymin(self):
        return min(o.ymin for o in self.textboxes)

    @property
    def xmax(self):
        return max(o.xmax for o in self.textboxes)

    @property
    def ymax(self):
        return max(o.ymax for o in self.textboxes)

    @property
    def width(self):
        return self.xmax - self.xmin

    @property
    def height(self):
        return self.ymax - self.ymin

    @property
    def x_center(self):
        return (self.xmax + self.xmin) / 2

    @property
    def y_center(self):
        return (self.ymax + self.ymin) / 2


class DocumentResult:
    def __init__(self) -> None:
        self.pages = []
        self.split_recommend = {}
        self.related = []
        self.all_tables = []
        self.results = {}
        self.formated_data = []