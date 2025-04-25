import sys
import threading
import time
import pyqtgraph as pg
from PyQt5.QtWidgets import (QApplication, QMainWindow, QVBoxLayout, QWidget, QMenu,
                             QScrollBar, QFileDialog, QInputDialog, QMenuBar, QAction)
from PyQt5.QtCore import Qt, QTimer
import numpy as np
from matplotlib import pyplot as plt
import librosa
from pydub import AudioSegment
from pydub.playback import play
import csv
from phoneme_segmenter import Audio  # <-- Import the backend Audio class

class Segment:
    '''
    This class is for SYLLABLE SEGMENTS, to be displayed directly on the spectrogram
    '''
    def __init__(self, start_time, end_time, label, window):
        self.start_time = start_time
        self.end_time = end_time
        self.label = label
        self.region = pg.LinearRegionItem(values=(start_time, end_time), movable=True)
        self.text_item = pg.TextItem(text=label, anchor=(0.5, 0))
        self.window = window # to play the audio loaded on the main window


class WordSegment:
    '''
    Class for WORD SEGMENTS to be displayed on a separate plot
    '''
    def __init__(self, start_time, end_time, label, window):
        self.start_time = float(start_time)
        self.end_time = float(end_time)
        self.label = label
        self.window = window

        self.line_start = pg.InfiniteLine(pos=self.start_time, angle=90, pen=pg.mkPen('y', width=1))
        self.line_end = pg.InfiniteLine(pos=self.end_time, angle=90, pen=pg.mkPen('y', width=1))
        self.text_item = pg.TextItem(text=label, anchor=(0.5, 0.5))
        self.midpoint = (self.start_time + self.end_time) / 2
        self.text_item.setPos(self.midpoint, 0.5)

        self.add_to_plot()

    def add_to_plot(self):
        self.window.word_plot.addItem(self.line_start)
        self.window.word_plot.addItem(self.line_end)
        self.window.word_plot.addItem(self.text_item)

    def remove_from_plot(self):
        self.window.word_plot.removeItem(self.line_start)
        self.window.word_plot.removeItem(self.line_end)
        self.window.word_plot.removeItem(self.text_item)

    def update_position(self, start_time, end_time):
        self.start_time = float(start_time)
        self.end_time = float(end_time)
        self.line_start.setValue(self.start_time)
        self.line_end.setValue(self.end_time)
        self.midpoint = (self.start_time + self.end_time) / 2
        self.text_item.setPos(self.midpoint, 0.5)

class MainWindow(QMainWindow):
    def __init__(self, audio_file=None, syll_csv_file=None):
        super().__init__()
        self.audio_data = None
        self.sr = None
        self.pydub_audio = None
        self.spectrogram_data = None
        self.max_time = 0
        self.segments = []
        self.word_segments = []
        self.audio_obj = None
        self.energy_curve = None
        self.local_minima_lines = []
        
        if audio_file:
            self.load_audio(audio_file)
        if syll_csv_file:
            self.load_syll_csv(syll_csv_file)
            
        self.init_ui()
        self.init_menu()
        
    def load_audio(self, filename):
        self.audio_data, self.sr = librosa.load(filename, sr=None)
        self.spectrogram_data, self.max_time = self.compute_spectrogram()
        # pydub audio
        self.pydub_audio = AudioSegment.from_wav(filename)
        # backend audio object
        self.audio_obj = Audio(filename.split('/')[-1])

    def play_segment(self, start, end):
        # convert to ms
        start_ms, end_ms = int(1000*start), int(1000*end)
        play(self.pydub_audio[start_ms:end_ms])


    def compute_spectrogram(self):
        n_fft = 2048
        hop_length = 128
        
        stft = librosa.stft(self.audio_data, n_fft=n_fft, hop_length=hop_length)
        spectrogram = np.abs(stft)
        spectrogram_db = librosa.amplitude_to_db(spectrogram, ref=np.max)
        times = librosa.times_like(spectrogram_db, sr=self.sr, 
                                 hop_length=hop_length, n_fft=n_fft)
        return spectrogram_db.T, times[-1]

        
    def init_ui(self):
        self.setWindowTitle("Audio Segment Annotator")
        self.setGeometry(100, 100, 800, 600)
        
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        layout = QVBoxLayout(central_widget)
        
        # Create graphics layout with spectrogram, energy and label plots
        self.graphics_layout = pg.GraphicsLayoutWidget()
        self.spectrogram_plot = self.graphics_layout.addPlot(row=0, col=0)
        self.word_plot = self.graphics_layout.addPlot(row=1, col=0)
        self.label_plot = self.graphics_layout.addPlot(row=2, col=0)
        self.energy_plot = self.graphics_layout.addPlot(row=3, col=0)

        self.label_plot.setMaximumHeight(100)
        self.word_plot.setMaximumHeight(100)

        # Configure plots
        self.spectrogram_plot.setLabel('bottom', 'Time', units='s')
        self.spectrogram_plot.setLabel('left', 'Frequency', units='Hz')
        self.energy_plot.setLabel('left', 'Energy')
        self.energy_plot.setMouseEnabled(y=False)
        self.word_plot.hideAxis('left')
        self.word_plot.setYRange(0, 1)
        self.word_plot.setMouseEnabled(y=False)
        self.label_plot.hideAxis('left')
        self.label_plot.setYRange(0, 1)
        self.label_plot.setMouseEnabled(y=False)
        self.spectrogram_plot.setXLink(self.energy_plot)
        self.energy_plot.setXLink(self.label_plot)
        self.word_plot.setXLink(self.spectrogram_plot)
        
        # Create and configure scroll bar
        self.scroll_bar = QScrollBar(Qt.Horizontal)
        self.scroll_bar.setMaximum(int(self.max_time - 3))
        self.scroll_bar.valueChanged.connect(self.update_plot_range)
        
        # Add items to layout
        layout.addWidget(self.graphics_layout)
        layout.addWidget(self.scroll_bar)
        
        # Display spectrogram
        if self.spectrogram_data is not None:
            img = pg.ImageItem(self.spectrogram_data)
            img.setRect(pg.QtCore.QRectF(0, 0, self.max_time, self.sr/2))
            colormap = plt.get_cmap("jet")
            lut = (colormap(np.linspace(0, 1, 256))[:, :3] * 255).astype(np.uint8)
            img.setLookupTable(lut)
            self.spectrogram_plot.addItem(img)
            self.spectrogram_plot.setXRange(0, 3)


        if self.audio_obj:
            energy_time, energy, _ = self.audio_obj.get_energy()
            self.energy_curve = self.energy_plot.plot(energy_time, energy, pen='r') if self.energy_plot else None
            # energy threshold line
            self.threshold_line = pg.InfiniteLine(pos=self.audio_obj.energy_thresh, angle=0, 
                                                  pen=pg.mkPen('g', width=1), movable=True)
            self.threshold_line.sigPositionChanged.connect(self.update_threshold)
            self.energy_plot.addItem(self.threshold_line)

        for segment in self.segments:
            self.add_segment(segment)
        
        self.label_plot.scene().sigMouseClicked.connect(self.on_label_clicked)
        self.spectrogram_plot.scene().sigMouseClicked.connect(self.on_scene_clicked)

    def on_scene_clicked(self, event):
        pos = event.scenePos()
        for segment in self.segments:
            view_coords = self.spectrogram_plot.vb.mapSceneToView(pos)
            x = view_coords.x()
            if segment.start_time <= x <= segment.end_time:
                if abs(segment.end_time - segment.start_time) > 0.1:
                    self.play_segment(segment.start_time, segment.end_time)
                else: #when playing short segments like phoneme, play the surrounding context
                    self.play_segment(segment.start_time-0.2, segment.end_time+0.2)
                    
                break
        
    def init_menu(self):
        menu_bar = self.menuBar()
        file_menu = menu_bar.addMenu("&File")
        process_menu = menu_bar.addMenu("&Process")

        open_audio_action = QAction("Open Audio...", self)
        open_audio_action.triggered.connect(self.open_audio)
        file_menu.addAction(open_audio_action)
        
        open_csv_action = QAction("Open CSV...", self)
        open_csv_action.triggered.connect(self.open_csv)
        file_menu.addAction(open_csv_action)

        open_word_csv_action = QAction("Open word CSV...", self)
        open_word_csv_action.triggered.connect(self.open_word_csv)
        file_menu.addAction(open_word_csv_action)
        
        save_csv_action = QAction("Save CSV...", self)
        save_csv_action.triggered.connect(self.save_csv)
        file_menu.addAction(save_csv_action)

        get_silences_action = QAction("Get Silences", self)
        get_silences_action.triggered.connect(self.display_silences)
        process_menu.addAction(get_silences_action)

        adjust_silences_action = QAction("Adjust Long Silences", self)
        adjust_silences_action.triggered.connect(self.adjust_long_silences)
        process_menu.addAction(adjust_silences_action)

        snap_boundaries_action = QAction("Snap Boundaries to Minima", self)
        snap_boundaries_action.triggered.connect(self.snap_boundaries)
        process_menu.addAction(snap_boundaries_action)

        smooth_energy_action = QAction("Smooth Energy", self)
        smooth_energy_action.triggered.connect(self.smooth_energy)
        menu_bar.addAction(smooth_energy_action)

        show_minima_action = QAction("Show Local Minima", self)
        show_minima_action.triggered.connect(self.show_local_minima)
        process_menu.addAction(show_minima_action)
        
        merge_short_syllables_action = QAction("Merge short syllables", self)
        merge_short_syllables_action.triggered.connect(self.merge_short_syllables)
        process_menu.addAction(merge_short_syllables_action)
        
        split_long_syllables_action = QAction("Split long syllables", self)
        split_long_syllables_action.triggered.connect(self.split_long_syllables)
        process_menu.addAction(split_long_syllables_action)

        ghetto_action = QAction("Ghetto", self)
        ghetto_action.triggered.connect(self.ghetto)
        process_menu.addAction(ghetto_action)

    def ghetto(self):
        if not self.audio_obj or not hasattr(self.audio_obj, 'word_seg'):
            return
        self.audio_obj.ghetto_whspr_syll_transcript()
        self.update_visual_segments()

    def display_word_segments(self):
        if not self.audio_obj or not hasattr(self.audio_obj, 'word_seg'):
            return

        self.clear_word_segments()
        for start, end, label in self.audio_obj.word_seg:
            word_segment = WordSegment(start, end, label, self)
            self.word_segments.append(word_segment)

    def clear_word_segments(self):
        for word_segment in self.word_segments:
            word_segment.remove_from_plot()
        self.word_segments = []


    def display_silences(self):
        if not self.audio_obj:
            return
        silences = self.audio_obj.get_silence()
        for start, end in silences:
            segment = Segment(start, end, 'silence', self)
            self.segments.append(segment)
            self.add_segment(segment)

    def adjust_long_silences(self):
        if not self.audio_obj:
            return
        self.audio_obj.adjust_long_silences()
        self.clear_word_segments()
        self.display_word_segments()

    def snap_boundaries(self):
        if not self.audio_obj:
            return
        choice, ok = QInputDialog.getItem(
            self, "Select Segments", "Apply snap to:", ["phoneme", "syllable", "word"], 0, False)
        if ok and choice:
            if choice == "syllable" and hasattr(self.audio_obj, 'syll_seg'):
                self.audio_obj.syll_seg = self.audio_obj.snap_close_boundaries_to_minimum(self.audio_obj.syll_seg)
                self.update_visual_segments()
            elif choice == "word" and hasattr(self.audio_obj, 'word_seg'):
                self.audio_obj.word_seg = self.audio_obj.snap_close_boundaries_to_minimum(self.audio_obj.word_seg)
                self.clear_word_segments()
                self.display_word_segments()
            elif choice == "phoneme" and hasattr(self.audio_obj, 'phon_seg'):
                self.audio_obj.word_seg = self.audio_obj.snap_close_boundaries_to_minimum(self.audio_obj.phon_seg)
                self.update_visual_segments()


    def smooth_energy(self):
        if not self.audio_obj:
            return
        self.audio_obj.smooth_energy()
        energy_time = self.audio_obj.energy_time
        energy = self.audio_obj.energy
        self.energy_curve.setData(energy_time, energy)
    
    def merge_short_syllables(self):
        if not self.audio_obj:
            return
        self.audio_obj.merge_short_syllables()
        self.update_visual_segments()
    
    def split_long_syllables(self):
        if not self.audio_obj:
            return
        self.audio_obj.split_long_syllables()
        self.update_visual_segments()

    def show_local_minima(self):
        if not self.audio_obj:
            return
        # Remove old lines
        for line in self.local_minima_lines:
            self.energy_plot.removeItem(line)
        self.local_minima_lines.clear()
 
        minima_times = self.audio_obj.get_local_minima()[1]

        for t in minima_times:
            line = pg.InfiniteLine(pos=t, angle=90, pen=pg.mkPen('b', width=1))
            self.energy_plot.addItem(line)
            self.local_minima_lines.append(line)

    def update_visual_segments(self):
        # Clear existing label plot segments
        for segment in self.segments:
            self.spectrogram_plot.removeItem(segment.region)
            self.label_plot.removeItem(segment.text_item)

        self.segments = []
        for start, end, label in self.audio_obj.syll_seg:
            segment = Segment(start, end, label, self)
            self.segments.append(segment)
            self.add_segment(segment)

    def add_segment(self, segment):
        self.spectrogram_plot.addItem(segment.region)
        segment.region.sigRegionChanged.connect(
            lambda: self.on_region_changed(segment))
        self.label_plot.addItem(segment.text_item)
        self.update_text_position(segment)
        
    def on_region_changed(self, segment):
        start, end = segment.region.getRegion()
        segment.start_time = start
        segment.end_time = end
        self.update_text_position(segment)
        
    def update_text_position(self, segment):
        midpoint = (segment.start_time + segment.end_time) / 2
        segment.text_item.setPos(midpoint, 0.5)
        
    def on_label_clicked(self, event):
        if event.double():
            items = self.label_plot.scene().items(event.scenePos())
            for item in items:
                if isinstance(item, pg.TextItem):
                    self.edit_label(item)
                    break
                    
    def edit_label(self, text_item):
        for segment in self.segments:
            if segment.text_item is text_item:
                new_text, ok = QInputDialog.getText(
                    self, "Edit Label", "New label:", text=segment.label)
                if ok:
                    segment.label = new_text
                    text_item.setText(new_text)
                break
                
    def load_syll_csv(self, filename):
        self.segments = []
        self.audio_obj.syll_seg = []
        with open(filename, 'r') as f:
            reader = csv.reader(f)
            for row in reader:
                try:
                    start = float(row[0])
                    end = float(row[1])
                    label = row[2]
                    self.segments.append(Segment(start, end, label, self))
                    self.audio_obj.syll_seg.append((start, end, label))
                except:
                    continue
    
    def open_word_csv(self):
        filename, _ = QFileDialog.getOpenFileName(
            self, "Open CSV File", "", "CSV Files (*.csv)")
        if filename:
            self.audio_obj.csv_to_word_seg(filename)
            self.display_word_segments()
            #self.init_ui()

    def update_threshold(self):
        if self.audio_obj and hasattr(self, 'threshold_line'):
            self.audio_obj.energy_thresh = self.threshold_line.value()
                    
    def save_csv(self):
        filename, _ = QFileDialog.getSaveFileName(
            self, "Save CSV", "", "CSV Files (*.csv)")
        if filename:
            with open(filename, 'w', newline='') as f:
                writer = csv.writer(f)
                for segment in self.segments:
                    writer.writerow([segment.start_time, 
                                    segment.end_time, 
                                    segment.label])
                                    
    def open_audio(self):
        filename, _ = QFileDialog.getOpenFileName(
            self, "Open Audio File", "", "Audio Files (*.wav *.mp3)")
        if filename:
            self.load_audio(filename)
            self.init_ui()
            
    def open_csv(self):
        '''
        to open syllable CSVs
        '''
        filename, _ = QFileDialog.getOpenFileName(
            self, "Open CSV File", "", "CSV Files (*.csv)")
        if filename:
            self.load_syll_csv(filename)
            #self.init_ui()
            self.update_visual_segments()

            
    def update_plot_range(self):
        value = self.scroll_bar.value()
        self.spectrogram_plot.setXRange(value, value + 3)
        self.energy_plot.setXRange(value, value + 3)
        self.label_plot.setXRange(value, value + 3)
        
if __name__ == "__main__":
    app = QApplication(sys.argv)
    main = MainWindow()
    main.show()
    sys.exit(app.exec_())

