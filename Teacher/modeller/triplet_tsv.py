from collections import OrderedDict
from Animator.utils import string_to_pil

try:
    try:
        # noinspection PyCompatibility
        from cStringIO import StringIO
    except ImportError:
        from io import StringIO
except ImportError:
    # noinspection PyCompatibility
    from StringIO import StringIO


class TripletTsv(object):
    def __init__(self, path):
        self._path = path
        self._data = {}
        self._len = None
        self.__new_w = 256
        self.__new_h = 256
        self._load_data(path)

    def __repr__(self):
        return 'TripletTsv(size: {} path: {})'.format(
            len(self), self._path
        )

    def __len__(self):
        return self._len

    def _load_data(self, path):
        """Load source ranges
        :param path: path to triplet.tsv
        """
        self._data = OrderedDict()
        with open(path) as fp:
            for index, im64 in enumerate(fp):
                anc_pos_neg = im64.split('\t')
                self._data[index] = [string_to_pil(im_b64[1:]) for im_b64 in anc_pos_neg]
                self._len = index

    def __getitem__(self, index):
        """Convert to the normalized tuple key to be used to look up in index
        :rtype: tuple[str, str, str]
        """
        # key is flat index
        anc, pos, neg = self._data[index]
        return anc, pos, neg

    def __iter__(self):
        for index, (anc, pos, neg) in enumerate(self._data):
            yield index, (anc, pos, neg)
