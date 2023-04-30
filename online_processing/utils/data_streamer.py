from data_acquisition.modules.headset.EEG import EEG

headset = EEG()


def empty_buffer():
    """
    Clear headset buffer (should be called once at the beginning of processing)
    """
    headset.clear_tasks()


def get_epoc(w, fs):
    return headset.epoc_streamer(w, fs)
