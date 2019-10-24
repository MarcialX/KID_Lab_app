#!/usr/bin/env python
# -*- coding: utf-8 -*-
# The homodyne system for AIG lab
#
# Copyright (C) November, 2018  Becerril, Marcial <mbecerrilt@inaoep.mx>
# Author: Becerril, Marcial <mbecerrilt@inaoep.mx> based in the codes of
# Pete Barry et al. (pcp), Sam Gordon <sbgordo1@asu.edu>, Sam Rowe and
# Thomas Gascard.
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.

import os
import os.path
import sys
import time
import numpy as np
from numpy import fft
import struct
from socket import *
from scipy import signal, ndimage, fftpack
import logging

import threading

from scipy.signal import savgol_filter

import casperfpga
import pygetdata as gd

from IPython.qt.console.rich_ipython_widget import RichIPythonWidget
from IPython.qt.inprocess import QtInProcessKernelManager

from threading import Event

from PyQt4 import QtCore, QtGui,uic
from PyQt4.QtCore import *
from PyQt4.QtGui import QPalette,QWidget,QFileDialog,QMessageBox, QTreeWidgetItem, QIcon, QPixmap, QTableWidgetItem

import matplotlib.pyplot as plt
from astropy.io import fits
import matplotlib
import matplotlib.gridspec as gridspec
from matplotlib.figure import Figure
from matplotlib.backends.backend_qt4agg import(
    FigureCanvasQTAgg as FigureCanvas,
    NavigationToolbar2QT as NavigationToolbar)

from PyQt4.QtCore import QThread, SIGNAL

from config_window import configWindow

sys.path.append('./misc/')
import color_msg_html as cm

# The core of the program PCP -> Pete Barry et al. Rifado el Pete
import pcp

import plot_sweep_avg

# Wait N seconds to get the time stream
class waitWhileStream(QThread):

    def __init__(self, roach, time, ntimes):
        QtCore.QThread.__init__(self)
        self.roach = roach
        self.time = time
        self.ntimes = ntimes + 1
        self.finished = Event()

    def run(self):
        cnt = 1
        self.roach.start_stream()
        while not self.finished.wait(self.time):
            self.roach.stop_stream()
            cnt += 1

            self.emit(SIGNAL('val_pbar(float)'), cnt/float(self.ntimes))
            self.emit(SIGNAL('dirfile(QString)'), self.roach.writer_daemon.current_filename)
            if cnt > 1 and cnt < self.ntimes:
                self.roach.start_stream()
            
            if cnt > self.ntimes:
                self.cancel()

    def cancel(self):
        self.finished.set()


#Write target tones
class thread_target_tones(QThread):

    def __init__(self,roach,freqs,amps,phases,type_tones):
        QThread.__init__(self)
        self.roach = roach
        self.freqs = freqs
        self.amps = amps
        self.phases = phases

    def __del__(self):
        self.wait()

    def run(self):

        try:
            self.roach.roach_iface.write_freqs_to_qdr(self.freqs, self.amps, self.phases)
            self.emit(SIGNAL('status(int)'), 0)
        except:
            self.emit(SIGNAL('status(int)'), 1)

#Write target tones
class thread_vna_tones(QThread):

    def __init__(self,roach,freqs,amps,phases,type_tones):
        QThread.__init__(self)
        self.roach = roach
        self.freqs = freqs
        self.amps = amps
        self.phases = phases

    def __del__(self):
        self.wait()

    def run(self):

        try:
            self.roach.roach_iface.write_freqs_to_qdr(self.freqs, self.amps, self.phases)
            self.emit(SIGNAL('status(int)'), 0)
        except:
            self.emit(SIGNAL('status(int)'), 1)

#Write test comb
class thread_write_test(QThread):

    def __init__(self,roach,freqs,amps,phases, type_tones):
        QThread.__init__(self)
        self.roach = roach
        self.freqs = freqs
        self.amps = amps
        self.phases = phases

    def __del__(self):
        self.wait()

    def run(self):
        try:
            self.roach.roach_iface.write_freqs_to_qdr(self.freqs, self.amps, self.phases)
            self.emit(SIGNAL('status(int)'), 0)
        except:
            self.emit(SIGNAL('status(int)'), 1)

#QDR Calibration
class qdr_cal_thread(QThread):

    def __init__(self, roach):
        QThread.__init__(self)
        self.roach = roach

    def __del__(self):
        self.wait()

    def run(self):
        fpga = self.roach.roach_iface.fpga
        if fpga.is_running():
            QDR = self.roach.roach_iface.calibrate_qdr()
            # QDR Calibration
            if QDR < 0:
                self.emit(SIGNAL('status(int)'), 0)
            else:
                pcp.lib.lib_fpga.write_to_fpga_register(fpga, { 'write_qdr_status_reg': 1 }, self.roach.roach_iface.firmware_reg_list )
                self.emit(SIGNAL('status(int)'), 1)
        else:
            self.emit(SIGNAL('status(int)'), 2)


#Target Sweep Thread
class tar_sweep_thread(QThread):

    def __init__(self, roach, Navg, startidx, stopidx, span, step):
        QThread.__init__(self)

        self.roach = roach
        self.Navg = Navg
        self.sweep_startidx = startidx
        self.sweep_stopidx = stopidx

        self.sweep_span = span
        self.sweep_step = step

    def __del__(self):
        self.wait()

    def run(self):
        
        step_times = []
        sleeptime = np.round( self.Navg / 488. * 1.05, decimals = 3 )#self.fpga.sample_rate) * 1.05 # num avgs / sample_rate + 5%

        self.roach.pre_sweep_lo(sweep_span=self.sweep_span, sweep_step=self.sweep_step)

        nTones = float(len(self.roach.toneslist.sweep_lo_freqs))
        try:
            cnt = 0
            for lo_freq in self.roach.toneslist.sweep_lo_freqs:
                self.roach.synth_lo.frequency = int(lo_freq)

                self.emit(SIGNAL('freq(float)'), lo_freq/1.0e6)
                self.emit(SIGNAL('fraction(float)'), cnt/nTones)
                cnt += 1

                step_times.append( time.time() )
                time.sleep(sleeptime)
        except KeyboardInterrupt:
            pass

        self.roach.post_sweep_lo(step_times=step_times, startidx=self.sweep_startidx, stopidx=self.sweep_stopidx)
        # Back to the central frequency
        self.roach.synth_lo.frequency = self.roach.toneslist.lo_freq
        self.emit(SIGNAL('freq(float)'), lo_freq/1.0e6)
        self.emit(SIGNAL('dirfile(QString)'), self.roach.current_sweep_dirfile.name)       

#VNA Sweep Thread
class vna_sweep_thread(QThread):

    def __init__(self, roach, Navg, startidx, stopidx, span, step):
        QThread.__init__(self)

        self.roach = roach
        self.Navg = Navg
        self.sweep_startidx = startidx
        self.sweep_stopidx = stopidx

        self.sweep_span = span
        self.sweep_step = step

    def __del__(self):
        self.wait()

    def run(self):
        self.roach.sweep_lo(sweep_span=self.sweep_span, sweep_step=self.sweep_step, sweep_avgs=self.Navg, startidx=self.sweep_startidx, stopidx=self.sweep_stopidx)

# Logging
class QTextEditLogger(logging.Handler):
    def __init__(self,parent):
        logging.Handler.__init__(self)
        self.widget = QtGui.QTextEdit(parent)
        self.widget.setReadOnly(True)

    def emit(self, record):
        msg = self.format(record)
        #self.widget.setStyleSheet("QTextEdit {background-color: #333;font-family: Courier;}")
        self.widget.append(msg)

    def write(self, m):
        pass

# Embed IPython terminal
class EmbedIPython(RichIPythonWidget):

    def __init__(self, **kwarg):
        super(RichIPythonWidget, self).__init__()
        self.kernel_manager = QtInProcessKernelManager()
        self.kernel_manager.start_kernel()
        self.kernel = self.kernel_manager.kernel
        self.kernel.gui = 'qt4'
        self.kernel.shell.push(kwarg)
        self.kernel_client = self.kernel_manager.client()
        self.kernel_client.start_channels()

# Main Window class
class MainWindow(QtGui.QMainWindow):
    def __init__(self):
        QtGui.QMainWindow.__init__(self)

        # Load of main window GUI
        # The GUI was developed in QT Designer
        self.ui = uic.loadUi("src/gui/main.ui")

        # Full Screen
        self.ui.showMaximized()
        screen = QtGui.QDesktopWidget().screenGeometry()

        # Screen dimensions
        self.size_x = screen.width()
        self.size_y = screen.height()

        self.ui.setWindowFlags(self.ui.windowFlags() | QtCore.Qt.CustomizeWindowHint)
        self.ui.setWindowFlags(self.ui.windowFlags() & ~QtCore.Qt.WindowMaximizeButtonHint)

        self.ui.ctrlFrame.resize(250,self.size_y - 145)
        self.ui.tabPlots.resize(self.size_x - self.ui.Terminal.width() - self.ui.ctrlFrame.width() - 30, self.size_y - 145)

        self.ui.plotPSDtimeStream.resize(self.ui.tabPlots.width() - self.ui.timeFrame.width() - 25, self.ui.tabPlots.height()/2 - 25)
        self.ui.plotPSDtimeStream.setLayout(self.ui.PSDPlot)

        self.ui.timeStreamLabel.move(10, self.ui.plotPSDtimeStream.height() + self.ui.plotPSDtimeStream.x() + 30)
        self.ui.plotTimeStream.resize(self.ui.tabPlots.width() - 20,self.ui.tabPlots.height()/2 - 75)
        self.ui.plotTimeStream.move(10, self.ui.plotPSDtimeStream.height() + self.ui.plotPSDtimeStream.x() + 50)
        self.ui.plotTimeStream.setLayout(self.ui.timeStreamPlot)

        self.ui.timeFrame.move(self.ui.plotPSDtimeStream.width() + 15, 50)
        self.ui.timeFrame.resize(180, self.ui.plotPSDtimeStream.height() - 20)
        self.ui.timeTree.resize(self.ui.timeFrame.width() - 10, self.ui.timeFrame.height() - 10)
        self.ui.writefitsfileTS.move(self.ui.timeFrame.x() + self.ui.timeFrame.width()/2 - self.ui.writefitsfileTS.width()/2, 20)       

        self.ui.plotVNAFrame.resize(self.ui.tabPlots.width() - self.ui.VNAFrame.width() - 25, 3*self.ui.tabPlots.height()/4 + 85)
        self.ui.plotVNAFrame.setLayout(self.ui.VNAPlot)

        self.ui.VNAFrame.move(self.ui.plotVNAFrame.width() + 15, 50)
        self.ui.VNAFrame.resize(180, self.ui.plotVNAFrame.height() - 20)
        self.ui.plotsVNATree.resize(self.ui.VNAFrame.width() - 10, self.ui.VNAFrame.height() - 10)
        self.ui.writefitsfileVNA.move(self.ui.VNAFrame.x() + self.ui.VNAFrame.width()/2 - self.ui.writefitsfileVNA.width()/2, 20)

        self.ui.plotHomoFrame.resize(self.ui.tabPlots.width() - self.ui.kidsFrame.width() - 25,self.ui.tabPlots.height()/2 - 25)
        self.ui.plotHomoFrame.setLayout(self.ui.HomoPlot)

        self.ui.iqCircleLabel.move(10, self.ui.plotHomoFrame.height() + self.ui.plotHomoFrame.x() + 30)
        self.ui.plotIQFrame.resize(self.ui.tabPlots.width()/2 - 20,self.ui.tabPlots.height()/2 - 75)
        self.ui.plotIQFrame.move(10,self.ui.plotHomoFrame.height() + self.ui.plotHomoFrame.x() + 50)
        self.ui.plotIQFrame.setLayout(self.ui.IQPlot)

        self.ui.speedLabel.move(self.ui.plotIQFrame.width() + 25, self.ui.plotHomoFrame.height() + self.ui.plotHomoFrame.x() + 30)
        self.ui.plotSpeedFrame.resize(self.ui.tabPlots.width()/2 - 20,self.ui.tabPlots.height()/2 - 75)
        self.ui.plotSpeedFrame.move(self.ui.plotIQFrame.width() + 25,self.ui.plotHomoFrame.height() + self.ui.plotHomoFrame.x() + 50)
        self.ui.plotSpeedFrame.setLayout(self.ui.SpeedPlot)

        self.ui.kidsFrame.move(self.ui.plotHomoFrame.width() + 15, 50)
        self.ui.kidsFrame.resize(180, self.ui.plotHomoFrame.height() - 20)
        self.ui.plotsTree.resize(self.ui.kidsFrame.width() - 10, self.ui.kidsFrame.height() - 10)
        self.ui.writefitsfileSweep.move(self.ui.kidsFrame.x() + self.ui.kidsFrame.width()/2 - self.ui.writefitsfileSweep.width()/2, 20)

        self.ui.Terminal.move(self.ui.tabPlots.width() + self.ui.ctrlFrame.width() + 20, self.size_y/2 - 25)
        #self.ui.Terminal.setLayout(self.ui.TermVBox)

        self.ui.loggsFrame.move(self.ui.tabPlots.width() + self.ui.ctrlFrame.width() + 20,10)
        self.ui.loggsFrame.resize(self.ui.Terminal.width(), self.size_y/2 - 80)
        self.ui.loggsFrame.setLayout(self.ui.loggsText)

        self.ui.progressBar.move(self.ui.tabPlots.width() + self.ui.ctrlFrame.width() + 20, self.size_y/2 - 62)
        self.ui.progressBar.resize(self.ui.Terminal.width(),30)

        # TAB DATA
        # Data
        self.ui.dataFrame.move(5,(self.ui.tabPlots.height() - self.ui.dataFrame.height() - 15)/2)
        self.ui.tableToneList.resize(self.ui.tabPlots.width() - self.ui.tableToneList.x() - 40, self.ui.tabPlots.height()-150)

        self.path_temp_tones = './tonelists/temp_tones.txt'
        self.path_vna_tones = './tonelists/tones_VNA.txt'

        # Logging
        logTextBox = QTextEditLogger(self)
        # You can format what is printed to text box
        #logTextBox.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
        logging.getLogger().addHandler(logTextBox)
        # You can control the logging level
        logging.getLogger().setLevel(logging.INFO)
        self.ui.loggsText.addWidget(logTextBox.widget)

        logTextBox.widget.setStyleSheet("QTextEdit {background-color: #DDD;font: 9pt Courier}")

        # Initial settings
        # Loading General Configuration file
        self.config = pcp.configuration
        roach_list = self.config.roach_config.keys()   

        # - - - - - - - - - - - - ROACH Settings - - - - - - - - - - - - - - - -
        # Roach
        self.ui.roach_ID.clear()
        self.ui.roach_ID.addItems(roach_list)

        # Create an object per each ROACH
        self.nRoach = {}
        for i in roach_list:
            self.nRoach[i] = {"obj":pcp.mux_channel.muxChannel(i)}
            self.nRoach[i]["synth"] = False
            self.nRoach[i]["atten"] = False

        # Everything is initialise in the first ROACH selected
        self.updateRoachFields(roach_list[0])

        # Network settings
        self.updateNetworkFields(roach_list[0])

        # Hardware settings
        # Synthesizer
        self.updateSynthFields(roach_list[0])
        # Attenuator
        self.updateAttFields(roach_list[0])

        # Write test comb
        self.updateWriteTestFields(roach_list[0])

        self.updateTonesFields(roach_list[0])
        self.updateTonesTable(roach_list[0])

        # Data Settings
        self.updateDataSet()

        # Tool bar
        # ROACH status
        self.ui.actionRoach.triggered.connect(self.roach_connection)
        # ROACH network
        self.ui.actionNetwork.triggered.connect(self.roach_network)
        # CHeck Packets received
        self.ui.actionPackets_received.triggered.connect(self.check_packets)
        # QDR Calibration
        self.ui.actionQDR_Calibration.triggered.connect(self.qdr_cal)
        # Synthesizer
        self.ui.actionSynthesizer.triggered.connect(self.roach_synth)
        # Attenuattors
        self.ui.actionInit_Attenuators.triggered.connect(self.roach_atten)
        # Attenuator calibration
        self.ui.actionADC_Calibration.triggered.connect(self.cal_atten)
        # Initialize all
        self.ui.actionInitialize.triggered.connect(self.init_all)
        self.ui.actionInitialise_Hardware.triggered.connect(self.init_hardware)

        # Configuration File
        self.ui.actionConfiguration_toolbar.triggered.connect(self.openConfigWindow)

        # Menu Bar
        # --> File
        self.ui.actionExit.triggered.connect(self.exitKIDLab)

        # --> Settings
        self.ui.actionConfiguration_menu.triggered.connect(self.openConfigWindow)

        # --> Roach
        self.ui.actionInitialize_ROACH.triggered.connect(self.init_all)
        self.ui.actionUpload_Firmware.triggered.connect(self.upload_firmware)
        self.ui.actionROACH_board.triggered.connect(self.roach_connection)

        self.ui.actionUDP_Configuration.triggered.connect(self.roach_network)
        self.ui.actionTest_UDP_Connection.triggered.connect(self.test_udp)

        self.ui.actionQDR_Calibration_menu.triggered.connect(self.qdr_cal)

        # --> Hardware
        self.ui.actionInitialize_synths.triggered.connect(self.roach_synth)
        self.ui.actionInitialize_attenuators.triggered.connect(self.roach_atten)

        #self.ui.actionRelocate_tones.triggered.connect(self.relocate_tones)
        self.ui.actionMax_Speed.triggered.connect(self.maxSpeed)
        self.ui.actionMinimum_S21.triggered.connect(self.minS21)
        self.ui.actionWrite_tones.triggered.connect(self.moveTones)

        # --> Help
        self.ui.actionAbout.triggered.connect(self.about)


        # Buttons
        # Roach
        # Roach Settings
        self.ui.roach_ID.currentIndexChanged.connect(self.changeActiveRoach)

        self.ui.firmDir.mousePressEvent = self.chooseFirmPath
        self.ui.saveDiryBtn.mousePressEvent = self.chooseSavePath
        self.ui.tonesDiryBtn.mousePressEvent = self.chooseTonesPath
        self.ui.firmDiryBtn.mousePressEvent = self.chooseFirmDiryPath

        self.ui.upFirmBtn.mousePressEvent = self.upload_firmware
        self.ui.synthBtn.mousePressEvent = self.roach_synth
        self.ui.udpConfBtn.mousePressEvent = self.roach_network
        self.ui.udpTestBtn.mousePressEvent = self.test_udp

        self.ui.setCLKBtn.mousePressEvent = self.setCLKFreq
        self.ui.setPowCLKBtn.mousePressEvent = self.setCLKPow

        self.ui.setLOBtn.mousePressEvent = self.setLOFreq
        self.ui.setPowLOBtn.mousePressEvent = self.setLOPow

        self.ui.attenBtn.mousePressEvent = self.roach_atten
        self.ui.setInAttBtn.mousePressEvent = self.setIn
        self.ui.setOutAttBtn.mousePressEvent = self.setOut

        self.ui.inAttenSet.valueChanged.connect(self.setInAttenModule)
        self.ui.outAttenSet.valueChanged.connect(self.setOutAttenModule)

        self.ui.attenCalBtn.mousePressEvent = self.cal_atten

        self.ui.updateTonesListBtn.mousePressEvent = self.saveChangesToneList

        self.ui.writeTestBtn.mousePressEvent = self.write_Tones_test

        self.ui.writeTonesBtn.mousePressEvent = self.write_Target_Tones

        self.ui.targetSweepBtn.mousePressEvent = self.target_Sweep
        self.ui.targetAddSweepBtn.mousePressEvent = self.add_target_Sweep

        self.ui.noiseTimeStreamBtn.mousePressEvent = self.newNoise
        self.ui.addNoiseTimeStreamBtn.mousePressEvent = self.addNoise

        self.ui.writeVNABtn.mousePressEvent = self.write_VNA_tones

        self.ui.newVNABtn.mousePressEvent = self.vna_Sweep
        self.ui.addVNABtn.mousePressEvent = self.add_vna_Sweep


        self.ui.locateResBtn.mousePressEvent = self.vnalocateRes
        #self.ui.saveTonesBtn.mousePressEvent = self.saveTones
        #self.ui.actionSave_settings.triggered.connect(self.saveSettings)

        self.ui.tonesBtn.mousePressEvent = self.chooseTargTones

        self.ui.actionShow_Tones.triggered.connect(self.showSweepTones)

        self.ui.rootDiryBtn.mousePressEvent = self.chooseRootPath
        self.ui.filePlotBtn.mousePressEvent = self.choosePlotTargPath

        self.ui.pathVNABtn.mousePressEvent = self.chooseVNAPath
        self.ui.filePlotTimeStreamBtn.mousePressEvent = self.chooseStreamPath

        # Plot Sweep
        self.ui.plotFileBtn.mousePressEvent = self.newSweepPlot
        self.ui.plotAddFileBtn.mousePressEvent = self.addSweepPlot

        self.ui.plotsTree.itemClicked.connect(self.plotSweepSelected)

        # Plot VNA
        self.ui.plotVNABtn.mousePressEvent = self.newVNAPlot
        self.ui.addPlotVNABtn.mousePressEvent = self.addVNAPlot

        self.ui.plotsVNATree.itemClicked.connect(self.plotVNASelected)

        # Plot Time Stream
        self.ui.plotFileTimeStreamBtn.mousePressEvent = self.newTimeStreamPlot
        self.ui.addPlotFileTimeStreamBtn.mousePressEvent = self.addTimeStreamPlot

        self.ui.timeTree.itemClicked.connect(self.plotTimeStreamSelected)

        # Iniatialising
        self.statusConn = 0
        self.statusFirm = 0
        self.statusSynth = 0
        self.statusAtt = 0
        self.statusNet = 0

        self.freqVNA = []
        self.magVNA = []
        self.ind = []

        self.flagVNA = False
        self.spanVNA = 0

        self.sweepData = {}
        self.vnaData = {}
        self.timeData = {}

        # Check Initial connections
        self.check_roach_connection()
        #self.check_packets_received()

        # Creation of Plots Sweep
        self.figHomo_sweep = Figure()
        self.addmpl_homodyne_sweep(self.figHomo_sweep)

        self.figHomo_iq = Figure()
        self.addmpl_homodyne_iq(self.figHomo_iq)

        self.figHomo_speed = Figure()
        self.addmpl_homodyne_speed(self.figHomo_speed)

        # Creation of Plot VNA
        self.figVNA = Figure()
        self.addmpl_vna(self.figVNA)

        # Plots Time Stream
        self.figTimeStream = Figure()
        self.addmpl_ts(self.figTimeStream)

        self.figPSD_ts = Figure()
        self.addmpl_ts_PSD(self.figPSD_ts)

        # To use LATEX in plots
        matplotlib.rc('text', usetex=True)
        matplotlib.rcParams['text.latex.preamble']=[r"\usepackage{amsmath}"]

        # IPython console
        #self.console = EmbedIPython()
        #self.console.kernel.shell.run_cell('%pylab qt')
        #self.console.execute("cd ./")
        #self.ui.TermVBox.addWidget(self.console)

        self.ui.progressBar.setValue(0)

        self.ui.show()

    def check_roach_connection(self):
        """
        Check if the roach is connected, running and the firmware is loaded
        """
        icon = QIcon()

        roach_status = self.testConn(self.ui.roach_ID.currentText())

        if not roach_status:
            self.statusConn = 0
            icon.addPixmap(QPixmap('./src/icon/wrong_icon.png'))
            self.ui.actionRoach_Status.setIcon(icon)
            self.ui.statusbar.showMessage(u'ROACH connection failed!')
            logging.warning(cm.WARNING + 'ROACH connection failed.' + cm.ENDC)

            self.ui.upFirmBtn.setStyleSheet("""QWidget {
                        color: white;
                        background-color: red
                        }""")

        else:
            self.statusConn = 1
            icon.addPixmap(QPixmap('./src/icon/ok_icon.png'))
            self.ui.actionRoach_Status.setIcon(icon)
            self.ui.statusbar.showMessage(u'ROACH connection is successful!')
            logging.info(cm.OK + 'ROACH connection is successful!' + cm.ENDC)

            # Check firmware
            if self.nRoach[self.ui.roach_ID.currentText()]["obj"].roach_iface.fpga:
                # If firmware is uploaded
                if self.nRoach[self.ui.roach_ID.currentText()]["obj"].roach_iface.fpg_uploaded:
                    self.ui.upFirmBtn.setStyleSheet("""QWidget {
                        color: white;
                        background-color: green
                        }""")
                    self.ui.statusbar.showMessage(u'Firmware already loaded')
                    logging.info(cm.INFO + 'Firmware already loaded.' + cm.ENDC)

            # if self.nRoach[self.ui.roach_ID.currentText()]["obj"].writer_daemon.check_packets_received():
            #     self.ui.actionPackets_received_Status.setIcon(icon)
            #     self.ui.actionNetwork_status.setIcon(icon)
            #     self.ui.actionQDR_Status.setIcon(icon)

            #     self.ui.udpConfBtn.setStyleSheet("""QWidget {
            #                             color: white;
            #                             background-color: green
            #                             }""")
            #     self.ui.udpTestBtn.setStyleSheet("""QWidget {
            #                             color: white;
            #                             background-color: green
            #                             }""")
            #     self.ui.statusbar.showMessage(u'Test successful!')
            #     logging.info(cm.OK + "Connections are working. Receiving packets!" + cm.ENDC)


    def check_packets_received(self):
        """
        Check if packages are already received
        """
        icon = QIcon()
        if not self.nRoach[self.ui.roach_ID.currentText()]["obj"].writer_daemon.check_packets_received():
            icon.addPixmap(QPixmap('./src/icon/wrong_icon.png'))
            self.ui.actionPackets_received_Status.setIcon(icon)
            self.ui.udpConfBtn.setStyleSheet("""QWidget {
                                    color: white;
                                    background-color: red
                                    }""")
            self.ui.udpTestBtn.setStyleSheet("""QWidget {
                                    color: white;
                                    background-color: red
                                    }""")
            self.ui.statusbar.showMessage(u'Error receiving data.')
            logging.warning(cm.ERROR + "Error receiving data. Check ethernet configuration." + cm.ENDC)

        else:
            icon.addPixmap(QPixmap('./src/icon/ok_icon.png'))
            self.ui.actionPackets_received_Status.setIcon(icon)
            self.ui.udpConfBtn.setStyleSheet("""QWidget {
                                    color: white;
                                    background-color: green
                                    }""")
            self.ui.udpTestBtn.setStyleSheet("""QWidget {
                                    color: white;
                                    background-color: green
                                    }""")
            self.ui.statusbar.showMessage(u'Test successful!')
            logging.info(cm.OK + "Connections are working. Receiving packets!" + cm.ENDC)


    def changeActiveRoach(self, index):
        new_roach_selected = self.ui.roach_ID.currentText()

        self.updateRoachFields(new_roach_selected)
        self.updateNetworkFields(new_roach_selected)

        self.updateSynthFields(new_roach_selected)
        self.updateAttFields(new_roach_selected)
        self.updateWriteTestFields(new_roach_selected)

        self.updateTonesFields(new_roach_selected)
        self.updateTonesList(new_roach_selected)
        self.updateTonesTable(new_roach_selected)

        self.updateDataSet()

    def openConfigWindow(self):
        roach_config = self.config.roach_config
        network_config = self.config.network_config
        filesys_config = self.config.filesys_config
        hardware_config = self.config.hardware_config
        logging_config = self.config.logging_config

        self.uiconfig = configWindow(self, roach_config, network_config, filesys_config, hardware_config, logging_config)
        self.uiconfig.saveRoachSig.connect(self.updateRoachFile)
        self.uiconfig.saveNetSig.connect(self.updateNetFile)
        self.uiconfig.saveHardSig.connect(self.updateHardFile)
        self.uiconfig.saveDirySig.connect(self.updateDiryFile)
        #self.uiconfig.saveLoggs.connect(self.updateLoggsFile)

        self.uiconfig.showGUI()

    def updateRoachFile(self, roachFile):
        self.config.roach_config = roachFile
        roach_selected = self.ui.roach_ID.currentText()

        self.updateRoachFields(roach_selected)

        self.updateSynthFields(roach_selected)
        self.updateAttFields(roach_selected)

        self.updateWriteTestFields(roach_selected)

        self.updateTonesFields(roach_selected)
        self.updateTonesList(roach_selected)
        self.updateTonesTable(roach_selected)

    def updateNetFile(self, netFile):
        self.config.network_config = netFile
        roach_selected = self.ui.roach_ID.currentText()

        self.updateNetworkFields(new_roach_selected)

    def updateHardFile(self, hardFile):
        self.config.hardware_config = hardFile
        roach_selected = self.ui.roach_ID.currentText()

        self.updateSynthFields(new_roach_selected)
        self.updateAttFields(new_roach_selected)

    def updateDiryFile(self, diryFile):
        self.config.filesys_config = diryFile
        roach_selected = self.ui.roach_ID.currentText()

        self.updateRoachFields(roach_selected)
        self.updateDataSet()

    """
    def updateLoggsFile(self, loggsFile):
        print loggsFile
    """

    def updateRoachFields(self, roachID):
        self.ui.roachIPEdit.setText(self.config.network_config[roachID]["roach_ppc_ip"])
        self.ui.firmEdit.setText(self.config.roach_config[roachID]["firmware_file"])

        # File system
        self.ui.savedatadiryEdit.setText(self.config.filesys_config["savedatadir"])
        self.ui.tonesdataEdit.setText(self.config.filesys_config["tonelistdir"])
        self.ui.firmdataEdit.setText(self.config.filesys_config["firmwaredir"])

    def updateNetworkFields(self, roachID):
        self.ui.ethportEdit.setText(self.config.network_config[roachID]["udp_dest_device"])
        self.ui.bufLenEdit.setText(str(self.config.network_config[roachID]["buf_size"]))
        self.ui.headLenEdit.setText(str(self.config.network_config[roachID]["header_len"]))

        # Source V6
        self.ui.ipSrcEdit.setText(self.config.network_config[roachID]["udp_source_ip"])
        self.ui.macSrcEdit.setText(self.config.network_config[roachID]["udp_source_mac"])
        self.ui.portSrcEdit.setText(str(self.config.network_config[roachID]["udp_source_port"]))

        # Destiny (Computer)
        self.ui.ipDstEdit.setText(self.config.network_config[roachID]["udp_dest_ip"])
        self.ui.macDstEdit.setText(self.config.network_config[roachID]["udp_dest_mac"])
        self.ui.portDstEdit.setText(str(self.config.network_config[roachID]["udp_dest_port"]))

        # Ethernet port
        os.system("sudo ip link set " + self.config.network_config[roachID]["udp_dest_device"] + " mtu 9000")

    def updateSynthFields(self, roachID):
        #First check which synthesizers are already defined in the configuration way
        self.ui.modelCLKBox.clear()
        self.ui.modelLOBox.clear()
        synthList = []
        for synth in self.config.hardware_config["synth_config"]:
            if self.config.hardware_config["synth_config"][synth]["active"]:
                synthList.append(synth)

        self.ui.modelCLKBox.addItems(synthList)
        self.ui.modelLOBox.addItems(synthList)

        i = 0
        for clk in synthList:
            if clk == self.config.roach_config[roachID]["synthid_clk"]:
                break
            i += 1

        self.ui.modelCLKBox.setCurrentIndex(i)

        j = 0
        for lo in synthList:
            if lo == self.config.roach_config[roachID]["synthid_lo"]:
                break
            j += 1

        self.ui.modelLOBox.setCurrentIndex(j)

    def updateAttFields(self, roachID):
        #First check which synthesizers are already defined in the configuration way
        self.ui.attInputBox.clear()
        self.ui.attOutputBox.clear()
        attList = []
        for att in self.config.hardware_config["atten_config"]:
            if self.config.hardware_config["atten_config"][att]["active"]:
                attList.append(att)

        self.ui.attInputBox.addItems(attList)
        self.ui.attOutputBox.addItems(attList)

        i = 0
        for input in attList:
            if input == self.config.roach_config[roachID]["att_in"]:
                break
            i += 1

        self.ui.attInputBox.setCurrentIndex(i)

        j = 0
        for out in attList:
            if out == self.config.roach_config[roachID]["att_out"]:
                break
            j += 1

        self.ui.attOutputBox.setCurrentIndex(j)

    def updateWriteTestFields(self, roachID):
        self.ui.minNegEdit.setText(str(self.config.roach_config[roachID]["min_neg_freq"]))
        self.ui.maxNegEdit.setText(str(self.config.roach_config[roachID]["max_neg_freq"]))

        self.ui.minPosEdit.setText(str(self.config.roach_config[roachID]["min_pos_freq"]))
        self.ui.maxPosEdit.setText(str(self.config.roach_config[roachID]["max_pos_freq"]))

        self.ui.offsetEdit.setText(str(self.config.roach_config[roachID]["symm_offset"]))
        self.ui.nFreqsEdit.setText(str(self.config.roach_config[roachID]["Nfreq"]))

    def updateTonesFields(self, roachID):
        self.ui.tonesEdit.setText(str(self.nRoach[roachID]["obj"].toneslist.tonelistfile))

        self.ui.spanTargEdit.setText(str(self.config.roach_config[roachID]["sweep_span"]))
        self.ui.stepTargEdit.setText(str(self.config.roach_config[roachID]["sweep_step"]))
        self.ui.avgsTargEdit.setText(str(self.config.roach_config[roachID]["sweep_avgs"]))

    def updateTonesTable(self, roachID):
        """
        Update Tones list Table
        """
        toneslist = self.nRoach[roachID]["obj"].toneslist.data

        self.ui.tableToneList.setRowCount(len(toneslist))

        for i in range(len(toneslist)):
            # Set table
            self.ui.tableToneList.setItem(i,0, QTableWidgetItem(str(toneslist['name'][i])))
            self.ui.tableToneList.setItem(i,1, QTableWidgetItem(str(toneslist['freq'][i])))
            #self.ui.tableToneList.setItem(i,2, QTableWidgetItem(str(toneslist['power'][i])))
            self.ui.tableToneList.setItem(i,2, QTableWidgetItem(str(toneslist['offset att'][i])))
            self.ui.tableToneList.setItem(i,3, QTableWidgetItem(str(toneslist['all'][i])))
            self.ui.tableToneList.setItem(i,4, QTableWidgetItem(str(toneslist['none'][i])))

        self.ui.tableToneList.resizeColumnsToContents()

    def updateDataSet(self):
        self.ui.rootFolderEdit.setText(str(self.config.filesys_config["rootdir"]))

    def cal_atten(self,event):
        """
        Function to calibrate the attenuation levels to use the full range of the ADC
        """
        print "In construction..."

    def init_all(self, event):
        """
        Function to initialize everything
        """
        w = QWidget()
        icon = QIcon()
        self.ui.setEnabled(False)
        QMessageBox.information(w, "Roach initialization", "Initailising Roach, from the connections to the Data Packets transmission")
        logging.info(cm.INFO + 'Initialising everything, from the Roach Connection to the Data Packets transmission.' + cm.ENDC)
        self.ui.statusbar.showMessage(u'Initialising everything...')
        icon.addPixmap(QPixmap('./src/icon/ok_icon.png'))

        roachID = self.ui.roach_ID.currentText()

        try:
            pps_active = self.ui.actionActive_PPS.isChecked()
            self.nRoach[roachID]["obj"].roach_iface.initialise_fpga(force_reupload = True, pps_active=pps_active)
            icon.addPixmap(QPixmap('./src/icon/ok_icon.png'))
            self.ui.actionNetwork_status.setIcon(icon)
            self.ui.actionQDR_Status.setIcon(icon)

        except:
            logging.info(cm.WARNING + 'Initialization error. Check roach connection' + cm.ENDC)
            self.ui.statusbar.showMessage(u'Initialization error. Check roach connection')
            QMessageBox.warning(w, "Initialization", "Error during the initialization of the ROACH, check connections.")

        self.ui.setEnabled(True)

    def init_hardware(self, event):
        """
        Function to initialize all the hardware
        """
        icon = QIcon()
        logging.info(cm.INFO + 'Initailising hardware devices: attenuators and synthesizers' + cm.ENDC)
        self.ui.statusbar.showMessage(u'Initialising hardware...')

        roachID = self.ui.roach_ID.currentText()

        self.clkFreq = np.float(self.ui.freqClk.text())*1e6
        self.clkPow = np.float(self.ui.powClk.text())
        self.LOFreq = np.float(self.ui.loFreq.text())*1e6
        self.LOPow = np.float(self.ui.loPow.text())

        self.attIn = np.float(self.ui.attInEdit.text())
        self.attOut = np.float(self.ui.attOutEdit.text())

        try:
            self.nRoach[self.ui.roach_ID.currentText()]["synth"] = True
            self.nRoach[self.ui.roach_ID.currentText()]["atten"] = True

            self.nRoach[roachID]["obj"]._initialise_synth_lo()
            self.nRoach[roachID]["obj"]._initialise_synth_clk()
            logging.info(cm.OK + 'Initailising hardware: Synthesizers loaded!' + cm.ENDC)
            icon.addPixmap(QPixmap('./src/icon/ok_icon.png'))
            self.ui.actionSynthesizer_status.setIcon(icon)
            self.ui.synthBtn.setStyleSheet("""QWidget {
                                        color: white;
                                        background-color: green
                                        }""")

            self.nRoach[self.ui.roach_ID.currentText()]["obj"].synth_lo.frequency = self.LOFreq
            self.nRoach[self.ui.roach_ID.currentText()]["obj"].synth_lo.power = self.LOPow

            self.nRoach[self.ui.roach_ID.currentText()]["obj"].synth_clk.frequency = self.clkFreq
            self.nRoach[self.ui.roach_ID.currentText()]["obj"].synth_clk.power = self.clkPow

            self.ui.targfreqLCD.display(self.LOFreq/1.0e6)
            self.ui.targclkLCD.display(self.clkFreq/1.0e6)

            self.nRoach[roachID]["obj"]._initialise_atten_in()
            self.nRoach[roachID]["obj"]._initialise_atten_out()
            logging.info(cm.OK + 'Initailising hardware: Attenuattors loaded!' + cm.ENDC)
            icon.addPixmap(QPixmap('./src/icon/ok_icon.png'))
            self.ui.actionAttenuator_Status.setIcon(icon)
            self.ui.attenBtn.setStyleSheet("""QWidget {
                                        color: white;
                                        background-color: green
                                        }""")

            self.nRoach[self.ui.roach_ID.currentText()]["obj"].input_atten.attenuation = self.attIn
            self.nRoach[self.ui.roach_ID.currentText()]["obj"].output_atten.attenuation = self.attOut

            self.ui.inAttDisplay.display(self.attIn)
            self.ui.OutAttDisplay.display(self.attOut)

            self.ui.inAttenSet.setValue(self.attIn)
            self.ui.outAttenSet.setValue(self.attOut)

        except:
            icon.addPixmap(QPixmap('./src/icon/wrong_icon.png'))
            self.ui.actionSynthesizer_status.setIcon(icon)
            self.ui.synthBtn.setStyleSheet("""QWidget {
                                        color: white;
                                        background-color: red
                                        }""")
            icon.addPixmap(QPixmap('./src/icon/wrong_icon.png'))
            self.ui.actionAttenuator_Status.setIcon(icon)
            self.ui.attenBtn.setStyleSheet("""QWidget {
                                        color: white;
                                        background-color: red
                                        }""")
            logging.info(cm.WARNING + 'Initialising hardware error. Check hardware connection' + cm.ENDC)
            self.ui.statusbar.showMessage(u'Initialization error. Check hardware connection')
            self.nRoach[self.ui.roach_ID.currentText()]["synth"] = False
            self.nRoach[self.ui.roach_ID.currentText()]["atten"] = False

    def relocate_tones(self, event):
        print "Relocating tones ..."

    def saveChangesToneList(self, event):
        rowText = []
        for idx in self.ui.tableToneList.selectionModel().selectedRows():
            rw = idx.row()

            name = self.ui.tableToneList.item(rw, 0).text()
            freq = self.ui.tableToneList.item(rw, 1).text()
            offset = self.ui.tableToneList.item(rw, 2).text()
            att = self.ui.tableToneList.item(rw, 3).text()
            all_field = self.ui.tableToneList.item(rw, 4).text()
            none = self.ui.tableToneList.item(rw, 5).text()

            rowText.append([name, freq, offset, att, all_field, none])

        if rowText == []:
            print "Seleccionar todo!"
        else:
            header = "name\tfreq\tpower\toffset\tatt\tall\tnone\n"

            file = open(self.path_temp_tones,'w')
            file.write(header)

            for line in rowText:
                row = line[0] + '\t' + line[1] + '\t' + line[2] + '\t' + line[3] + '\t' + line[4] + '\t' + line[5] + '\t' + line[6] + '\n'
                file.write(row)

            file.close()

            self.nRoach[self.ui.roach_ID.currentText()]["obj"].toneslist.load_tonelist(self.path_temp_tones)

    def check_packets(self, event):
        self.test_udp(event)

    def saveTones(self,event):
        w = QWidget()
        w.resize(320, 240)
        w.setWindowTitle("Select directory to save the Tone List")
        if len(self.ind) > 0:
            fileName = QFileDialog.getSaveFileName(self, 'Save Tone List', './', selectedFilter='*.txt')
            file = open(fileName, "w")

            # Write Tone List
            file.write("Name\tFreq\tOffset\tatt\tAll\tNone\n")

            for i in range(len(self.ind[m])):
                file.write("K"+'{:03d}'.format(i)+"\t"+str(self.freqVNA[i])+"\t"+"0"+"\t"+"1"+"\t"+"0\n")
            file.close()
        else:
            self.ui.statusbar.showMessage(u'There is not tones to write')
            logging.info(cm.INFO + 'There is not tones to write' + cm.ENDC)
            QMessageBox.information(w, "Not tones", "There is not tones to write.")

    def locateRes(self, event):
        pass

    def vnalocateRes(self,event):
        w = QWidget()
        if self.flagVNA == True:
            # Filter less order
            npoints = self.ui.nPointsFind.value()
            order = self.ui.orderFind.value()

            mag_wBL = savgol_filter(self.magVNA,npoints,order)
            mag_wBL = -1*mag_wBL

            if self.ui.AutoBtn.isChecked():
                # Parameters MPH and MPD
                mph = np.max(mag_wBL)/8
                mpd = len(self.freqVNA)/500

                self.ui.mphBox.setValue(mph)
                self.ui.mpdBox.setValue(mpd)

            else:
                # Parameters
                mph = self.ui.mphBox.value()
                mpd = self.ui.mpdBox.value()

            ind = detect_peaks(signal.detrend(mag_wBL),mph=mph,mpd=mpd)
            self.ui.statusText.append("KIDS = " + str(len(ind)) + " founded")

            # Store for resonators
            self.ind = ind

            for i in self.ind[m]:
                i = int(i)
                line = self.f1.axvline(self.freqVNA[i], linewidth=0.75)

            self.f1.figure.canvas.draw()

        else:
            self.ui.statusbar.showMessage(u'There is not VNA available')
            logging.info(cm.INFO + 'No VNA available' + cm.ENDC)
            QMessageBox.information(w, "No VNA", "It is not possible to get the resonators, there is not VNA available.")

    def saveSettings(self,event):
        print "Saving configuration settings"

    def choosePath(self,flag):
        w = QWidget()
        w.resize(320, 240)
        w.setWindowTitle("Select directory where KID files are ")

        if flag == "firm":
            firmware = QFileDialog.getOpenFileName(self, "Select File")
            if not firmware == "":
                self.ui.firmEdit.setText(firmware)
                self.config.roach_config[self.ui.roach_ID.currentText()]["firmware_file"] = firmware
        elif flag == "savePath":
            savepath = QFileDialog.getExistingDirectory(self, "Select directory")
            if not savepath == "":
                self.ui.savedatadiryEdit.setText(savepath)
                self.config.filesys_config["savedatadir"] = savepath
        elif flag == "tonesPath":
            tones_savepath = QFileDialog.getExistingDirectory(self, "Select directory")
            if not tones_savepath == "":
                self.ui.tonesdataEdit.setText(tones_savepath)
                self.config.filesys_config["tonelistdir"] = tones_savepath
        elif flag == "firmDiryPath":
            dirfile_savepath = QFileDialog.getExistingDirectory(self, "Select directory")
            if not dirfile_savepath == "":
                self.ui.firmdataEdit.setText(dirfile_savepath)
                self.config.filesys_config["firmwaredir"] = dirfile_savepath
        elif flag == "tonesList":
            toneslist_savepath = QFileDialog.getOpenFileName(self, "Select File")
            if not toneslist_savepath == "":
                self.ui.tonesEdit.setText(toneslist_savepath)
                self.config.roach_config[self.ui.roach_ID.currentText()]["tonelist_file"] = toneslist_savepath
                roachID = self.ui.roach_ID.currentText()
                self.updateTonesList(roachID)
                self.updateTonesTable(roachID)
        elif flag == "rootPath":
            root_path = QFileDialog.getExistingDirectory(self, "Select Folder")
            if not root_path == "":
                self.ui.rootFolderEdit.setText(root_path)
                self.config.filesys_config["rootPath"] = root_path
        elif flag == "plotTargPath":
            plot_path = QFileDialog.getExistingDirectory(self, "Select Folder")
            if not plot_path == "":
                self.ui.plotSweepEdit.setText(plot_path)
        elif flag == "vnaPath":
            plot_vna_path = QFileDialog.getExistingDirectory(self, "Select Folder")
            if not plot_vna_path == "":
                self.ui.plotVNAEdit.setText(plot_vna_path)
        elif flag == "streamPath":
            plot_stream_path = QFileDialog.getExistingDirectory(self, "Select Folder")
            if not plot_stream_path == "":
                self.ui.plotTimeStreamEdit.setText(plot_stream_path)

    def chooseFirmPath(self, event):
        self.choosePath("firm")

    def chooseSavePath(self, event):
        self.choosePath("savePath")

    def chooseTonesPath(self, event):
        self.choosePath("tonesPath")

    def chooseFirmDiryPath(self, event):
        self.choosePath("firmDiryPath")

    def chooseTargTones(self, event):
        self.choosePath("tonesList")

    def chooseRootPath(self, event):
        self.choosePath("rootPath")

    def choosePlotTargPath(self, event):
        self.choosePath("plotTargPath")

    def chooseVNAPath(self, event):
        self.choosePath("vnaPath")

    def chooseStreamPath(self, event):
        self.choosePath("streamPath")

    def testConn(self, roach):
        """Tests the link to Roach2 PPC
            inputs:
                casperfpga object fpga: The fpga object
            outputs: the fpga object"""
        try:
            is_connected = self.nRoach[roach]["obj"].roach_iface.fpga.is_connected()
            if not is_connected:
                logging.warning(cm.WARNING + "No connection to ROACH. If booting, wait 30 seconds and retry. Otherwise, check gc config." + cm.ENDC)
        except RuntimeError:
            is_connected = False
            logging.warning(cm.WARNING + "No connection to ROACH. If booting, wait 30 seconds and retry. Otherwise, check gc config." + cm.ENDC)
        except:
            is_connected = False
            logging.warning(cm.WARNING + "No connection to ROACH." + cm.ENDC)
        return is_connected

    def roach_connection(self,event):
        """Check the connection with ROACH, if it is connected turn green the status icon"""

        roach_ip = self.ui.roachIPEdit.text()

        w = QWidget()
        icon = QIcon()
        self.ui.setEnabled(False)
        self.ui.statusbar.showMessage(u'Waiting for roach connection...')
        QMessageBox.information(w, "ROACH Connection", "Starting with ROACH comunication ...")

        roach_status = self.testConn(self.ui.roach_ID.currentText())

        if not roach_status:
            self.statusConn = 0
            icon.addPixmap(QPixmap('./src/icon/wrong_icon.png'))
            self.ui.actionRoach_Status.setIcon(icon)
            self.ui.statusbar.showMessage(u'ROACH connection failed!')
            logging.warning(cm.WARNING + 'ROACH connection failed.' + cm.ENDC)
            QMessageBox.information(w, "ROACH Connection", "No connection to ROACH. If booting, wait 30 seconds and retry. Otherwise, check gc config.")
        else:
            self.statusConn = 1
            icon.addPixmap(QPixmap('./src/icon/ok_icon.png'))
            self.ui.actionRoach_Status.setIcon(icon)
            self.ui.statusbar.showMessage(u'ROACH connection is successful!')
            logging.info(cm.OK + 'ROACH connection is successful!' + cm.ENDC)
            QMessageBox.information(w, "ROACH Connection", "Successful communication!")

        self.ui.setEnabled(True)

    def roach_synth(self,event):
        """Synthesizer connection. Check if the synthesizer is connected and set it
            the initial parameters"""

        # Show that it is already Initialized, in case of
        if self.nRoach[self.ui.roach_ID.currentText()]["synth"]:
            self.ui.synthBtn.setStyleSheet("""QWidget {
                                        color: white;
                                        background-color: green
                                        }""")
        else:
            self.ui.synthBtn.setStyleSheet("""QWidget {
                                        color: white;
                                        background-color: red
                                        }""")

        self.clkFreq = np.float(self.ui.freqClk.text())*1e6
        self.clkPow = np.float(self.ui.powClk.text())
        self.LOFreq = np.float(self.ui.loFreq.text())*1e6
        self.LOPow = np.float(self.ui.loPow.text())

        w = QMessageBox()
        icon = QIcon()

        FREQ_SYNTH_MAX = 5.e9
        FREQ_SYNTH_MIN = 54.e6

        POW_SYNTH_MIN = -30
        POW_SYNTH_MAX = 20

        assert self.clkFreq <= FREQ_SYNTH_MAX, QMessageBox.warning(w, "Clock frequency", "Clock Frequency is over range. For the ROACH it used to be 512 MHz")
        assert self.clkFreq >= FREQ_SYNTH_MIN, QMessageBox.warning(w, "Clock frequency", "Clock Frequency is under range. For the ROACH it used to be 512 MHz")

        assert self.LOFreq <= FREQ_SYNTH_MAX, QMessageBox.warning(w, "LO frequency", "LO Frequency is over range."+ str(FREQ_SYNTH_MAX) + " MHz is the highest limit")
        assert self.LOFreq >= FREQ_SYNTH_MIN, QMessageBox.warning(w, "LO frequency", "LO Frequency is under range."+ str(FREQ_SYNTH_MIN) +" MHz is the lowest limit")

        assert self.clkPow <= POW_SYNTH_MAX, QMessageBox.warning(w, "Clock Power", "Clock Power is over range."+ str(POW_SYNTH_MIN) + " dBm is the highest power")
        assert self.clkPow >= POW_SYNTH_MIN, QMessageBox.warning(w, "Clock Power", "Clock Power is under range."+ str(POW_SYNTH_MIN) + " dBm is the lowest power")

        assert self.LOPow <= POW_SYNTH_MAX, QMessageBox.warning(w, "LO Power", "LO Power is over range."+ str(POW_SYNTH_MIN) + " dBm is the highest power")
        assert self.LOPow >= POW_SYNTH_MIN, QMessageBox.warning(w, "LO Power", "LO POwer is under range."+ str(POW_SYNTH_MIN) + " dBm is the lowest power")

        self.ui.setEnabled(False)
        self.ui.statusbar.showMessage(u'Waiting for synthesizer connection ...')
        QMessageBox.information(w, "Synthesizer Connection", "Starting Synthesizer configuration ...")

        try:

            self.nRoach[self.ui.roach_ID.currentText()]["obj"]._initialise_synth_clk()
            self.nRoach[self.ui.roach_ID.currentText()]["obj"]._initialise_synth_lo()

            icon.addPixmap(QPixmap('./src/icon/ok_icon.png'))
            self.ui.actionSynthesizer_status.setIcon(icon)
            self.ui.synthBtn.setStyleSheet("""QWidget {
                                        color: white;
                                        background-color: green
                                        }""")
            self.nRoach[self.ui.roach_ID.currentText()]["synth"] = True
            logging.info(cm.OK + 'Synthesizer connection is successful' + cm.ENDC)
            self.ui.statusbar.showMessage(u'Synthesizer connection is successful')

            self.nRoach[self.ui.roach_ID.currentText()]["obj"].synth_lo.frequency = self.LOFreq
            self.nRoach[self.ui.roach_ID.currentText()]["obj"].synth_lo.power = self.LOPow

            self.nRoach[self.ui.roach_ID.currentText()]["obj"].synth_clk.frequency = self.clkFreq
            self.nRoach[self.ui.roach_ID.currentText()]["obj"].synth_clk.power = self.clkPow

            self.ui.targfreqLCD.display(self.LOFreq/1.0e6)
            self.ui.targclkLCD.display(self.clkFreq/1.0e6)

            QMessageBox.information(w, "Synthesizer connection", "Synthesizer connected and working!")
        except:
            icon.addPixmap(QPixmap('./src/icon/wrong_icon.png'))
            self.ui.actionSynthesizer_status.setIcon(icon)
            self.ui.synthBtn.setStyleSheet("""QWidget {
                                        color: white;
                                        background-color: red
                                        }""")
            self.nRoach[self.ui.roach_ID.currentText()]["synth"] = False
            logging.warning(cm.WARNING + 'Synthesizer failed!' + cm.ENDC)
            self.ui.statusbar.showMessage(u'Synthesizer failed!')
            QMessageBox.warning(w, "Synthesizer connection", "Synthesizer connection failed!")

        self.ui.setEnabled(True)

    def setCLKFreq(self, event):
        w = QMessageBox()
        icon = QIcon()

        if self.nRoach[self.ui.roach_ID.currentText()]["synth"]:
            self.clkFreq = np.float(self.ui.freqClk.text())*1e6
            self.nRoach[self.ui.roach_ID.currentText()]["obj"].synth_clk.frequency = self.clkFreq

            FREQ_SYNTH_MAX = 5.e9
            FREQ_SYNTH_MIN = 54.e6

            assert self.clkFreq <= FREQ_SYNTH_MAX, QMessageBox.warning(w, "Clock frequency", "Clock Frequency is over range. For the ROACH it used to be 512 MHz")
            assert self.clkFreq >= FREQ_SYNTH_MIN, QMessageBox.warning(w, "Clock frequency", "Clock Frequency is under range. For the ROACH it used to be 512 MHz")

            self.nRoach[self.ui.roach_ID.currentText()]["obj"].synth_clk.frequency = self.clkFreq
            self.ui.targclkLCD.display(self.clkFreq/1.0e6)

            logging.info(cm.INFO + 'CLK Frequency: '+ str(self.clkFreq) + cm.ENDC)
            self.ui.statusbar.showMessage('CLK Frequency: '+ str(self.clkFreq))
        else:
            logging.warning(cm.WARNING + 'Initialise the synthesizers first' + cm.ENDC)
            self.ui.statusbar.showMessage('Initialise the synthesizers first')

    def setCLKPow(self, event):
        w = QMessageBox()
        icon = QIcon()

        if self.nRoach[self.ui.roach_ID.currentText()]["synth"]:
            self.clkPow = np.float(self.ui.powClk.text())
            self.nRoach[self.ui.roach_ID.currentText()]["obj"].synth_clk.power = self.clkPow

            POW_SYNTH_MIN = -30
            POW_SYNTH_MAX = 20

            assert self.clkPow <= POW_SYNTH_MAX, QMessageBox.warning(w, "Clock Power", "Clock Power is over range."+ str(POW_SYNTH_MIN) + " dBm is the highest power")
            assert self.clkPow >= POW_SYNTH_MIN, QMessageBox.warning(w, "Clock Power", "Clock Power is under range."+ str(POW_SYNTH_MIN) + " dBm is the lowest power")

            self.nRoach[self.ui.roach_ID.currentText()]["obj"].synth_clk.power = self.clkPow

            logging.info(cm.INFO + 'CLK Power: '+ str(self.clkPow) + cm.ENDC)
            self.ui.statusbar.showMessage('CLK Power: '+ str(self.clkPow))
        else:
            logging.warning(cm.WARNING + 'Initialise the synthesizers first' + cm.ENDC)
            self.ui.statusbar.showMessage('Initialise the synthesizers first')

    def setLOFreq(self, event):
        w = QMessageBox()
        icon = QIcon()

        if self.nRoach[self.ui.roach_ID.currentText()]["synth"]:
            self.LOFreq = np.float(self.ui.loFreq.text())*1e6
            self.nRoach[self.ui.roach_ID.currentText()]["obj"].synth_lo.frequency = self.LOFreq

            FREQ_SYNTH_MAX = 5.e9
            FREQ_SYNTH_MIN = 54.e6

            assert self.LOFreq <= FREQ_SYNTH_MAX, QMessageBox.warning(w, "LO frequency", "LO Frequency is over range."+ str(FREQ_SYNTH_MAX) + " MHz is the highest limit")
            assert self.LOFreq >= FREQ_SYNTH_MIN, QMessageBox.warning(w, "LO frequency", "LO Frequency is under range."+ str(FREQ_SYNTH_MIN) +" MHz is the lowest limit")

            self.nRoach[self.ui.roach_ID.currentText()]["obj"].synth_lo.frequency = self.LOFreq
            self.ui.targfreqLCD.display(self.LOFreq/1.0e6)

            logging.info(cm.INFO + 'LO Frequency: '+ str(self.LOFreq) + cm.ENDC)
            self.ui.statusbar.showMessage('LO Frequency: '+ str(self.LOFreq))
        else:
            logging.warning(cm.WARNING + 'Initialise the synthesizers first' + cm.ENDC)
            self.ui.statusbar.showMessage('Initialise the synthesizers first')

    def setLOPow(self, event):
        w = QMessageBox()
        icon = QIcon()

        if self.nRoach[self.ui.roach_ID.currentText()]["synth"]:
            self.LOPow = np.float(self.ui.loPow.text())
            self.nRoach[self.ui.roach_ID.currentText()]["obj"].synth_lo.power = self.LOPow

            POW_SYNTH_MIN = -30
            POW_SYNTH_MAX = 20

            assert self.LOPow <= POW_SYNTH_MAX, QMessageBox.warning(w, "LO Power", "LO Power is over range."+ str(POW_SYNTH_MIN) + " dBm is the highest power")
            assert self.LOPow >= POW_SYNTH_MIN, QMessageBox.warning(w, "LO Power", "LO POwer is under range."+ str(POW_SYNTH_MIN) + " dBm is the lowest power")

            self.nRoach[self.ui.roach_ID.currentText()]["obj"].synth_lo.power = self.LOPow

            logging.info(cm.INFO + 'LO Power: '+ str(self.LOPow) + cm.ENDC)
            self.ui.statusbar.showMessage('LO Power: '+ str(self.LOPow))
        else:
            logging.warning(cm.WARNING + 'Initialise the synthesizers first' + cm.ENDC)
            self.ui.statusbar.showMessage('Initialise the synthesizers first')

    def qdrStatus(self,status):

        w = QMessageBox()
        icon = QIcon()

        if (status == 0):
            icon.addPixmap(QPixmap('./src/icon/wrong_icon.png'))
            self.ui.actionQDR_Status.setIcon(icon)
            self.ui.statusbar.showMessage(u'QDR Calibration failed!')
            logging.info(cm.WARNING + 'QDR Calibration failed!' + cm.ENDC)
            QMessageBox.information(w, "QDR Calibration", "QDR calibration failed... Check FPGA clock source")
        elif(status == 1):
            icon.addPixmap(QPixmap('./src/icon/ok_icon.png'))
            self.ui.actionQDR_Status.setIcon(icon)
            self.ui.statusbar.showMessage(u'QDR Calibration completed!')
            logging.info(cm.OK + 'QDR Calibration completed!' + cm.ENDC)
            QMessageBox.information(w, "QDR Calibration", "QDR calibration completed!")
        else:
            icon.addPixmap(QPixmap('./src/icon/wrong_icon.png'))
            self.ui.actionQDR_Status.setIcon(icon)
            logging.info(cm.WARNING + 'QDR calibration failed... Check ROACH connection' + cm.ENDC)
            QMessageBox.information(w, "QDR Calibration", "QDR calibration failed... Check ROACH connection")

    def qdr_cal(self,event):
        w = QMessageBox()

        self.thread_QDR = qdr_cal_thread(self.nRoach[self.ui.roach_ID.currentText()]["obj"])
        self.connect(self.thread_QDR, SIGNAL("status(int)"), self.qdrStatus)

        QMessageBox.information(w, "QDR Calibration", "Starting QDR calibration ...")
        self.ui.statusbar.showMessage(u'Waiting for QDR Calibration ...')
        logging.info(cm.INFO + "Starting QDR Calibration..." + cm.ENDC)

        self.thread_QDR.start()

    def roach_atten(self,event):
        """Attenuators connection. Check if the attenuators are connected and calibrate them"""

        # Show that it is already initialized
        if self.nRoach[self.ui.roach_ID.currentText()]["atten"]:
            self.ui.attenBtn.setStyleSheet("""QWidget {
                                        color: white;
                                        background-color: green
                                        }""")
        else:
            self.ui.attenBtn.setStyleSheet("""QWidget {
                                        color: white;
                                        background-color: red
                                        }""")

        self.attIn = np.float(self.ui.attInEdit.text())
        self.attOut = np.float(self.ui.attOutEdit.text())

        w = QMessageBox()
        icon = QIcon()

        self.ui.setEnabled(False)
        self.ui.statusbar.showMessage(u'Waiting for attenuator connection ...')
        QMessageBox.information(w, "Attenuator Connection", "Starting Attenuator configuration ...")

        try:
            self.nRoach[self.ui.roach_ID.currentText()]["obj"]._initialise_atten_in()
            self.nRoach[self.ui.roach_ID.currentText()]["obj"]._initialise_atten_out()

            icon.addPixmap(QPixmap('./src/icon/ok_icon.png'))
            self.ui.actionAttenuator_Status.setIcon(icon)
            self.ui.attenBtn.setStyleSheet("""QWidget {
                                        color: white;
                                        background-color: green
                                        }""")
            self.nRoach[self.ui.roach_ID.currentText()]["atten"] = True
            logging.info(cm.OK + 'Attenuator connection is successful' + cm.ENDC)
            self.ui.statusbar.showMessage(u'Attenuator connection is successful')

            self.nRoach[self.ui.roach_ID.currentText()]["obj"].input_atten.attenuation = self.attIn
            self.nRoach[self.ui.roach_ID.currentText()]["obj"].output_atten.attenuation = self.attOut

            self.ui.inAttDisplay.display(self.attIn)
            self.ui.OutAttDisplay.display(self.attOut)

            self.ui.inAttenSet.setValue(self.attIn)
            self.ui.outAttenSet.setValue(self.attOut)

            QMessageBox.information(w, "Attenuator connection", "Attenuators connected and working!")
        except Exception as e:
            icon.addPixmap(QPixmap('./src/icon/wrong_icon.png'))
            self.ui.actionAttenuator_Status.setIcon(icon)
            self.ui.attenBtn.setStyleSheet("""QWidget {
                                        color: white;
                                        background-color: red
                                        }""")
            self.nRoach[self.ui.roach_ID.currentText()]["atten"] = False
            logging.warning(cm.WARNING + 'Attenuator initialization failed!' + cm.ENDC)
            self.ui.statusbar.showMessage(u'Attenuator initialization failed!')
            QMessageBox.warning(w, "Attenuator connection", "Initialization of attenuators failed. " + str(e))

        self.ui.setEnabled(True)

    def setIn(self, event):
        self.attIn = np.float(self.ui.attInEdit.text())
        self.setInAtten(self.attIn)

    def setInAttenModule(self,event):
        self.attIn = self.ui.inAttenSet.value()
        self.setInAtten(self.attIn)

    def setInAtten(self, inAtt):
        if self.nRoach[self.ui.roach_ID.currentText()]["atten"]:

            self.nRoach[self.ui.roach_ID.currentText()]["obj"].input_atten.attenuation = inAtt
            self.ui.inAttDisplay.display(inAtt)
            self.ui.inAttenSet.setValue(inAtt)
            self.ui.attInEdit.setText(str(inAtt))

            logging.info(cm.INFO + 'Input Attenuation: '+ str(inAtt) + cm.ENDC)
            self.ui.statusbar.showMessage('Input Attenuation: '+ str(inAtt))
        else:
            logging.warning(cm.WARNING + 'Initialise the attenuators first' + cm.ENDC)
            self.ui.statusbar.showMessage('Initialise the attenuators first')

    def setOut(self,event):
        self.attOut = np.float(self.ui.attOutEdit.text())
        self.setOutAtten(self.attOut)

    def setOutAttenModule(self,event):
        self.attOut = self.ui.outAttenSet.value()
        self.setOutAtten(self.attOut)

    def setOutAtten(self, outAtt):
        if self.nRoach[self.ui.roach_ID.currentText()]["atten"]:

            self.nRoach[self.ui.roach_ID.currentText()]["obj"].output_atten.attenuation = outAtt
            self.ui.OutAttDisplay.display(outAtt)
            self.ui.outAttenSet.setValue(outAtt)
            self.ui.attOutEdit.setText(str(outAtt))

            logging.info(cm.INFO + 'Output Attenuation: '+ str(outAtt) + cm.ENDC)
            self.ui.statusbar.showMessage('Output Attenuation: '+ str(outAtt))
        else:
            logging.warning(cm.WARNING + 'Initialise the attenuators first' + cm.ENDC)
            self.ui.statusbar.showMessage('Initialise the attenuators first')

    def roach_network(self,event):

        self.config.network_config[self.ui.roach_ID.currentText()]['udp_dest_ip'] = self.ui.ipDstEdit.text()
        self.config.network_config[self.ui.roach_ID.currentText()]['udp_dst_port'] = self.ui.portDstEdit.text()
        self.config.network_config[self.ui.roach_ID.currentText()]['udp_dest_mac'] = self.ui.macDstEdit.text()

        self.config.network_config[self.ui.roach_ID.currentText()]['udp_src_ip'] = self.ui.ipSrcEdit.text()
        self.config.network_config[self.ui.roach_ID.currentText()]['udp_src_port'] = self.ui.portSrcEdit.text()
        self.config.network_config[self.ui.roach_ID.currentText()]['udp_src_mac'] = self.ui.macSrcEdit.text()

        self.config.network_config[self.ui.roach_ID.currentText()]['udp_dest_device'] = self.ui.ethportEdit.text()

        w = QMessageBox()
        icon = QIcon()

        self.ui.setEnabled(False)
        self.ui.statusbar.showMessage(u'UDP configuration ... ')
        QMessageBox.information(w, "UDP Configuration", "Starting UDP configuration ...")

        try:
            # UDP Configuration
            try:
                # configure downlink
                self.nRoach[self.ui.roach_ID.currentText()]["obj"].roach_iface.write_accum_len()
                self.nRoach[self.ui.roach_ID.currentText()]["obj"].roach_iface.configure_downlink_registers()

                icon.addPixmap(QPixmap('./src/icon/ok_icon.png'))
                self.ui.actionNetwork_status.setIcon(icon)
                self.ui.udpConfBtn.setStyleSheet("""QWidget {
                                        color: white;
                                        background-color: green
                                        }""")
                self.ui.statusbar.showMessage(u'UDP Downlink configured.')
                logging.info(cm.OK + 'UDP Downlink configured.' + cm.ENDC)
                QMessageBox.information(w, "UDP Downlink", "UDP Network configuraton id done.")

            except AttributeError:
                icon.addPixmap(QPixmap('./src/icon/wrong_icon.png'))
                self.ui.actionNetwork_status.setIcon(icon)
                self.ui.udpConfBtn.setStyleSheet("""QWidget {
                                        color: white;
                                        background-color: red
                                        }""")
                self.ui.statusbar.showMessage(u'UDP Downlink configuration failed!')
                logging.warning(cm.WARNING + "UDP Downlink could not be configured. Check ROACH connection." + cm.ENDC)
                QMessageBox.information(w, "UDP Downlink", "UDP Downlink could not be configured. Check ROACH connection.")

        except:
            icon.addPixmap(QPixmap('./src/icon/wrong_icon.png'))
            self.ui.actionNetwork_status.setIcon(icon)
            self.ui.udpConfBtn.setStyleSheet("""QWidget {
                                    color: white;
                                    background-color: red
                                    }""")
            self.statusNet = 1
            self.ui.statusbar.showMessage(u'UDP Network configuraton failed!')
            logging.warning(cm.WARNING + 'UDP Network configuraton failed!' + cm.ENDC)
            QMessageBox.information(w, "UDP error", "UDP Network configuraton failed! Check ROACH connection.")

        self.ui.setEnabled(True)

    def writeTestStatus(self,status):
        w = QMessageBox()
        icon = QIcon()

        if (status == 0):
            self.ui.statusbar.showMessage(u'Writing RF tones')
            logging.info(cm.INFO + "Writing RF tones" + cm.ENDC)
        if (status == 1):
            self.ui.statusbar.showMessage(u'Error finding dds shift')
            logging.warning(cm.WARNING + "Error finding dds shift: Try writing full frequency comb (N = 1000), or single test frequency. Then try again" + cm.ENDC)
        elif(status == 2):
            self.ui.statusbar.showMessage(u'QDR Calibration completed!')
            logging.info(cm.INFO + "Wrote DDS shift" + cm.ENDC)
        elif(status == 3):
            self.statusNet = 0
            self.ui.statusbar.showMessage("Ready! the tones are written")
            logging.info(cm.OK + "Tones written!" + cm.ENDC)
        elif(status == 4):
            self.statusNet = 1
            self.ui.statusbar.showMessage(u'Error writting test comb')
            logging.warning(cm.WARNING + 'Error writting test comb' + cm.ENDC)

    def test_udp(self, event):
        w = QMessageBox()
        icon = QIcon()

        self.ui.setEnabled(False)
        self.ui.statusbar.showMessage(u'Starting UDP test ... ')

        if not self.nRoach[self.ui.roach_ID.currentText()]["obj"].writer_daemon.check_packets_received():
            icon.addPixmap(QPixmap('./src/icon/wrong_icon.png'))
            self.ui.udpTestBtn.setStyleSheet("""QWidget {
                                    color: white;
                                    background-color: red
                                    }""")
            self.ui.statusbar.showMessage(u'Error receiving data.')
            logging.warning(cm.ERROR + "Error receiving data. Check ethernet configuration." + cm.ENDC)

        else:
            icon.addPixmap(QPixmap('./src/icon/ok_icon.png'))
            self.ui.udpTestBtn.setStyleSheet("""QWidget {
                                    color: white;
                                    background-color: green
                                    }""")
            self.ui.statusbar.showMessage(u'Test successful!')
            logging.info(cm.OK + "Test successful. Connections are working." + cm.ENDC)

        self.ui.actionPackets_received_Status.setIcon(icon)
        self.ui.actionNetwork_status.setIcon(icon)
        self.ui.actionQDR_Status.setIcon(icon)

        self.ui.setEnabled(True)

    def writeTargStatus(self,status,type_tones="any"):
        w = QMessageBox()

        if (status == 0):
            if type_tones == "vna":
                # Create a temporal toneslist just for the VNA files
                print "Creando archivo temporal"

            self.ui.statusbar.showMessage(u'Tones writen')
            logging.info(cm.OK + "Tones writen!" + cm.ENDC)
            QMessageBox.information(w, "Tones writen", "The tones has been written on the ROACH.")

        elif (status == 1):
            self.ui.statusbar.showMessage(u'Error writing tones.')
            logging.error(cm.ERROR + "Error writing tones. Check ROACH connection." + cm.ENDC)
            QMessageBox.critical(w, "Writing tones", "Error writing tones, check the connection with the ROACH.")

    def updateTonesList(self, roachID, file_changed=True):
        tonesList = self.ui.tonesEdit.text()
        self.config.roach_config[roachID]["tonelist_file"] = tonesList
        self.nRoach[self.ui.roach_ID.currentText()]["obj"].toneslist.load_tonelist(tonesList)

    def newNoise(self, event):
        roachID = self.ui.roach_ID.currentText()

        self.timeData = {}
        try:
            self.figTimeStream.clf()
            self.figPSD_ts.clf()
        except Exception as e:
            pass

        self.writeStream(roachID)

    def addNoise(self, event):
        roachID = self.ui.roach_ID.currentText()

        self.writeStream(roachID)

    def writeStream(self, roachID):
        w = QMessageBox()

        time = float(self.ui.spanTimeStreamEdit.text())
        ntimes = int(self.ui.nSamplesEdit.text())

        self.ui.statusbar.showMessage(u'Starting time stream...')
        logging.info(cm.INFO + "Starting time stream" + cm.ENDC)

        if not ntimes > 0:
            QMessageBox.information(w, "Starting time stream", "Number of samples has to be greater than 0")
            return

        try:
            self.ui.statusbar.showMessage(u'Taking time Stream...')

            roach = self.nRoach[roachID]["obj"]
            self.thread_wait_n_secs = waitWhileStream(roach, time, ntimes)
            self.connect(self.thread_wait_n_secs, SIGNAL("val_pbar(float)"), self.stop_write_stream)
            self.connect(self.thread_wait_n_secs, SIGNAL("dirfile(QString)"), self.timePlot)

            self.thread_wait_n_secs.start()
        except:
            self.ui.statusbar.showMessage(u'Error taking time streams.')
            logging.error(cm.ERROR + "Error taking time streams. Check ROACH connection and UDP packets adquisition" + cm.ENDC)
            QMessageBox.critical(w, "Time streams", "Check ROACH connection and UDP packets adquisition.")


    def stop_write_stream(self, value):
        self.set_progressBar(value)

    def write_Tones_test(self,event):
        w = QMessageBox()

        roachID = self.ui.roach_ID.currentText()

        # Read of the parameters
        min_neg = float(self.ui.minNegEdit.text())
        max_neg = float(self.ui.maxNegEdit.text())
        offset = float(self.ui.offsetEdit.text())

        min_pos = float(self.ui.minPosEdit.text())
        max_pos = float(self.ui.maxPosEdit.text())
        nFreqs = float(self.ui.nFreqsEdit.text())

        neg_freqs, neg_delta = np.linspace(min_neg + offset, max_neg + offset, nFreqs/2, retstep = True)
        pos_freqs, pos_delta = np.linspace(min_pos, max_pos, nFreqs/2, retstep = True)

        freq_comb = np.concatenate((neg_freqs, pos_freqs))
        freq_comb = freq_comb[freq_comb != 0]
        freq_comb = np.roll(freq_comb, - np.argmin(np.abs(freq_comb)) - 1)

        bb_freqs = np.sort(freq_comb)
        amps = np.ones_like(bb_freqs)
        phases = np.random.uniform(0,2*np.pi,len(amps))

        self.ui.statusbar.showMessage(u'Writting test comb ... ')
        QMessageBox.information(w, "Write the target tones", "Writting the target tones list ...")

        try:
            self.thread_write_test_comb = thread_write_test(self.nRoach[roachID]["obj"], bb_freqs, amps, phases, type_tones="test")
            self.connect(self.thread_write_test_comb, SIGNAL("status(int)"), self.writeTargStatus)

            self.thread_write_test_comb.start()

        except Exception as e:
            self.ui.statusbar.showMessage(u'Error writing tones.')
            logging.error(cm.ERROR + "Error writing tones. Check ROACH connection." + cm.ENDC)
            QMessageBox.critical(w, "Writing tones", "Error writing tones, check the connection with the ROACH.")

    def write_VNA_tones(self, event):
        w = QMessageBox()

        roachID = self.ui.roach_ID.currentText()

        # Read of the parameters
        Navg = int(self.ui.avgsTargEdit.text())

        center_freq = np.float(self.ui.centralEdit.text())*1.0e6
        startVNA = np.float(self.ui.startEdit.text())*1.0e6
        stopVNA = np.float(self.ui.stopEdit.text())*1.0e6
        lo_step_VNA = np.float(self.ui.stepEdit.text())
        Nfreq = int(self.ui.nTonesEdit.text())

        FREQ_SYNTH_MAX = 5.e9
        FREQ_SYNTH_MIN = 54.e6

        MAX_BW = np.float(self.config.roach_config[roachID]['dac_bandwidth'])

        assert center_freq <= FREQ_SYNTH_MAX, QMessageBox.warning(w, "Center frequency", "Center Frequency is over range")
        assert center_freq >= FREQ_SYNTH_MIN, QMessageBox.warning(w, "Center frequency", "Center Frequency is under range")
        assert stopVNA-startVNA > 0., QMessageBox.warning(w, "Frequency range", "The frequency range is not valid.")
        assert stopVNA-startVNA < MAX_BW, QMessageBox.warning(w, "Frequency range", "The frequency is over the bandwidth of the roach 512 MHz.")

        span = (stopVNA - startVNA)/Nfreq

        start = center_freq - (span)
        stop = center_freq + (span)

        sweep_freqs = np.arange(start, stop, lo_step_VNA)
        sweep_freqs = np.round(sweep_freqs/lo_step_VNA)*lo_step_VNA

        bb_freqs = np.linspace(startVNA, stopVNA, Nfreq)
        bb_freqs = bb_freqs - center_freq

        amps = np.ones_like(bb_freqs)
        phases = np.random.uniform(0,2*np.pi,len(amps))

        self.spanVNA = int(bb_freqs[1] - bb_freqs[0])

        tones = []

        cnt = 0
        for i in bb_freqs:
            row = ['T'+str(cnt).zfill(3), str(int(i + center_freq)), '0', '1', '0', ''] 
            tones.append(row)
            cnt += 1

        self.ui.statusbar.showMessage(u'Writting VNA tones ... ')
        QMessageBox.information(w, "Writing the VNA tones", "Writting the VNA tones list ...")

        try:
            self.writeTonesList(self.path_vna_tones, tones)
            self.nRoach[self.ui.roach_ID.currentText()]["obj"].toneslist.load_tonelist(self.path_vna_tones)

            self.thread_write_vna_tones = thread_vna_tones(self.nRoach[roachID]["obj"], bb_freqs, amps, phases, type_tones="vna")
            self.connect(self.thread_write_vna_tones, SIGNAL("status(int)"), self.writeTargStatus)

            self.thread_write_vna_tones.start()

        except Exception as e:
            self.ui.statusbar.showMessage(u'Error writing tones.')
            logging.error(cm.ERROR + "Error writing tones. Check ROACH connection." + cm.ENDC)
            QMessageBox.critical(w, "Writing tones", "Error writing tones, check the connection with the ROACH.")

    def writeTonesList(self, file, tones_dict):
        header = "name\tfreq\toffset\tatt\tall\tnone\n"

        file = open(file,'w')
        file.write(header)

        for line in tones_dict:
            row = line[0] + '\t' + line[1] + '\t' + line[2] + '\t' + line[3] + '\t' + line[4] + '\t' + line[5] + '\n'
            file.write(row)

        file.close()

    def write_Target_Tones(self,event):
        w = QMessageBox()

        roachID = self.ui.roach_ID.currentText()
        self.updateTonesList(roachID)

        self.ui.statusbar.showMessage(u'Writting test comb ... ')
        QMessageBox.information(w, "Write the target tones", "Writting the target tones list ...")

        if len(self.nRoach[roachID]["obj"].toneslist.bb_freqs) > 1:
            try:
                print self.nRoach[roachID]["obj"].toneslist.bb_freqs
                self.thread_write_targ_sweep = thread_target_tones(self.nRoach[roachID]["obj"], self.nRoach[roachID]["obj"].toneslist.bb_freqs, self.nRoach[roachID]["obj"].toneslist.amps, self.nRoach[roachID]["obj"].toneslist.phases, type_tones="target")
                self.connect(self.thread_write_targ_sweep, SIGNAL("status(int)"), self.writeTargStatus)

                self.thread_write_targ_sweep.start()

            except Exception as e:
                self.ui.statusbar.showMessage(u'Error writing tones.')
                logging.error(cm.ERROR + "Error writing tones. Check ROACH connection." + cm.ENDC)
                QMessageBox.critical(w, "Writing tones", "Error writing tones, check the connection with the ROACH.")
        else:
            self.ui.statusbar.showMessage(u'Error writing tones.')
            logging.error(cm.ERROR + "Error writing tones. One tone writting is not supported by the moment." + cm.ENDC)
            QMessageBox.critical(w, "Writing tones", "One tone writting is not supported by the moment.")

    def target_Sweep(self,event):
        roachID = self.ui.roach_ID.currentText()

        self.sweepData = {}
        try:
            self.figHomo_sweep.clf()
            self.figHomo_iq.clf()
            self.figHomo_speed.clf()
        except Exception as e:
            pass

        self.targetSweep(roachID)

    def add_target_Sweep(self,event):
        roachID = self.ui.roach_ID.currentText()
        self.targetSweep(roachID)

    def targetSweep(self, roachID):
        """
        Sweep function
        """
        w = QMessageBox()

        avgTxt =  self.ui.avgsTargEdit.text()
        try:
            if avgTxt == "Default":
                Navgs = 10
            else:
                Navgs = int(avgTxt)
            self.config.roach_config[roachID]["sweep_avgs"] = Navgs
        except:
            self.ui.statusbar.showMessage(u'Error in Sweeping.')
            logging.error(cm.ERROR + "Error in sweeping. Sweep Average number is not valid." + cm.ENDC)
            QMessageBox.critical(w, "Error in sweeping", "Sweep Average number is not valid.")
            return

        startTxt = self.ui.startidxTargEdit.text()
        try:
            if startTxt == "Default":
                startidx = 0
            else:
                startidx = int(startTxt)
        except:
            self.ui.statusbar.showMessage(u'Error in Sweeping.')
            logging.error(cm.ERROR + "Error in sweeping. Start index is not valid." + cm.ENDC)
            QMessageBox.critical(w, "Error in sweeping", "Start index is not valid.")
            return

        stopTxt = self.ui.stopidxTargEdit.text()
        try:
            if stopTxt == "Default":
                stopidx = Navgs
            else:
                stopidx = int(stopTxt)
        except:
            self.ui.statusbar.showMessage(u'Error in Sweeping.')
            logging.error(cm.ERROR + "Error in sweeping. Stop index is not valid." + cm.ENDC)
            QMessageBox.critical(w, "Error in sweeping", "Stop index is not valid.")
            return

        if (stopidx - startidx) < 0:
            self.ui.statusbar.showMessage(u'Error in Sweeping.')
            logging.error(cm.ERROR + "Error in sweeping. Start index is greater than stop index." + cm.ENDC)
            QMessageBox.critical(w, "Error in sweeping", "Start index is greater than stop index.")
            return

        spanSweep = self.ui.spanTargEdit.text()
        try:
            span = float(spanSweep)
            self.config.roach_config[roachID]["sweep_span"] = span
        except:
            self.ui.statusbar.showMessage(u'Error in Sweeping.')
            logging.error(cm.ERROR + "Error in sweeping. The frequency span is not valid." + cm.ENDC)
            QMessageBox.critical(w, "Error in sweeping", "The frequency span is not valid.")
            return

        stepSweep = self.ui.stepTargEdit.text()
        try:
            step = float(stepSweep)
            self.config.roach_config[roachID]["sweep_step"] = step
        except:
            self.ui.statusbar.showMessage(u'Error in Sweeping.')
            logging.error(cm.ERROR + "Error in sweeping. The frequency step is not valid." + cm.ENDC)
            QMessageBox.critical(w, "Error in sweeping", "The frequency step is not valid.")
            return

        try:
            roach = self.nRoach[roachID]["obj"]

            self.thread_targ_sweep = tar_sweep_thread(roach, Navgs, startidx, stopidx, span, step)
            self.connect(self.thread_targ_sweep, SIGNAL("freq(float)"), self.show_LO_Freq)
            self.connect(self.thread_targ_sweep, SIGNAL("fraction(float)"), self.set_progressBar)
            self.connect(self.thread_targ_sweep, SIGNAL("dirfile(QString)"), self.sweepPlot)

            self.thread_targ_sweep.start()
        except:
            self.ui.statusbar.showMessage(u'Error with the target sweep.')
            logging.error(cm.ERROR + "Target sweep error. Check ROACH and synthesizer connection." + cm.ENDC)
            QMessageBox.critical(w, "Target sweep error", "Error making the sweep, check the connection with the ROACH and synthesizer")

    def show_LO_Freq(self, value):
        self.ui.targfreqLCD.display(value)

    def vna_Sweep(self,event):
        roachID = self.ui.roach_ID.currentText()

        self.vnaData = {}
        try:
            self.figVNA.clf()
        except Exception as e:
            pass

        self.vnaSweep(roachID)

    def add_vna_Sweep(self,event):
        roachID = self.ui.roach_ID.currentText()
        self.vnaSweep(roachID)

    def vnaSweep(self, roachID):
        """
        VNA Sweep
        """

        w = QMessageBox()

        avgTxt =  self.ui.avgsTargEdit.text()
        try:
            if avgTxt == "Default":
                Navgs = 10
            else:
                Navgs = int(avgTxt)
            self.config.roach_config[roachID]["sweep_avgs"] = Navgs
        except:
            self.ui.statusbar.showMessage(u'Error in Sweeping.')
            logging.error(cm.ERROR + "Error in sweeping. Sweep Average number is not valid." + cm.ENDC)
            QMessageBox.critical(w, "Error in sweeping", "Sweep Average number is not valid.")
            return

        startTxt = self.ui.startidxTargEdit.text()
        try:
            if startTxt == "Default":
                startidx = 0
            else:
                startidx = int(startTxt)
        except:
            self.ui.statusbar.showMessage(u'Error in Sweeping.')
            logging.error(cm.ERROR + "Error in sweeping. Start index is not valid." + cm.ENDC)
            QMessageBox.critical(w, "Error in sweeping", "Start index is not valid.")
            return

        stopTxt = self.ui.stopidxTargEdit.text()
        try:
            if stopTxt == "Default":
                stopidx = Navgs
            else:
                stopidx = int(stopTxt)
        except:
            self.ui.statusbar.showMessage(u'Error in Sweeping.')
            logging.error(cm.ERROR + "Error in sweeping. Stop index is not valid." + cm.ENDC)
            QMessageBox.critical(w, "Error in sweeping", "Stop index is not valid.")
            return

        if (stopidx - startidx) < 0:
            self.ui.statusbar.showMessage(u'Error in Sweeping.')
            logging.error(cm.ERROR + "Error in sweeping. Start index is greater than stop index." + cm.ENDC)
            QMessageBox.critical(w, "Error in sweeping", "Start index is greater than stop index.")
            return

        stepSweep = self.ui.stepVNAEdit.text()
        try:
            step = float(stepSweep)
            self.config.roach_config[roachID]["sweep_step"] = step
        except:
            self.ui.statusbar.showMessage(u'Error in Sweeping.')
            logging.error(cm.ERROR + "Error in sweeping. The frequency step is not valid." + cm.ENDC)
            QMessageBox.critical(w, "Error in sweeping", "The frequency step is not valid.")
            return

        try:
            if self.spanVNA != 0:
                roach = self.nRoach[roachID]["obj"]
                self.thread_vna_sweep = tar_sweep_thread(roach, Navgs, startidx, stopidx, self.spanVNA, step)
                self.connect(self.thread_vna_sweep, SIGNAL("freq(float)"), self.show_LO_Freq)
                self.connect(self.thread_vna_sweep, SIGNAL("fraction(float)"), self.set_progressBar)
                self.connect(self.thread_vna_sweep, SIGNAL("dirfile(QString)"), self.vnaPlot)

                self.thread_vna_sweep.start()
            else:
                self.ui.statusbar.showMessage(u'Error with the target sweep.')
                logging.error(cm.WARNING + "Target sweep error. Write the tones first." + cm.ENDC)
                QMessageBox.critical(w, "Target sweep error", "Write the tones first. ")

        except Exception as e:
            self.ui.statusbar.showMessage(u'Error with the target sweep.')
            logging.error(cm.ERROR + "Target sweep error. Check ROACH and synthesizer connection." + cm.ENDC)
            QMessageBox.critical(w, "Target sweep error", "Error making the sweep. "+str(e))


    def upload_firmware(self,event):
        w = QWidget()

        self.ui.setEnabled(False)
        QMessageBox.information(w, "ROACH Connection", "Uploading firmware ...")

        firmware = self.ui.roachIPEdit.text()
        self.config.roach_config[self.ui.roach_ID.currentText()]["firmware_file"] = firmware

        try:
            self.nRoach[self.ui.roach_ID.currentText()]["obj"].roach_iface.upload_firmware_file(force_reupload=True)
            self.ui.upFirmBtn.setStyleSheet("""QWidget {
                                    color: white;
                                    background-color: green
                                    }""")
            QMessageBox.information(w, "ROACH Firmware", "Firmware uploaded successfuly! :)")
        except:
            QMessageBox.information(w, "ROACH Firmware", "Firmware upload failed! :(")
            self.ui.upFirmBtn.setStyleSheet("""QWidget {
                                    color: white;
                                    background-color: red
                                    }""")

        self.ui.setEnabled(True)

    def clearTree(self, tree):
        iterator = QtGui.QTreeWidgetItemIterator(tree, QtGui.QTreeWidgetItemIterator.All)
        while iterator.value():
            iterator.value().takeChildren()
            iterator +=1

        i = tree.topLevelItemCount()
        while i > -1:
            tree.takeTopLevelItem(i)
            i -= 1

    def updateSweepPlots(self):
        checked = dict()
        root = self.ui.plotsTree.invisibleRootItem()
        signal_count = root.childCount()
        rescale = False

        x_sweep_max = -np.inf
        x_sweep_min = np.inf

        y_sweep_max = -np.inf
        y_sweep_min = np.inf

        x_speed_max = -np.inf
        x_speed_min = np.inf

        y_speed_max = -np.inf
        y_speed_min = np.inf

        for i in range(signal_count):
            signal = root.child(i)
            checked_sweeps = list()
            num_children = signal.childCount()

            file = signal.text(0)
            for j in range(num_children):
                child = signal.child(j)
                if child.checkState(0) == QtCore.Qt.Checked:
                    checked_sweeps.append(child.text(0))

                    self.sweepData[file]["sweep"][j][0].set_visible(True)
                    self.sweepData[file]["iq"][j][0].set_visible(True)
                    self.sweepData[file]["speed"]["signal"][j][0].set_visible(True)

                    if self.ui.actionShow_Tones.isChecked():
                        self.sweepData[file]["points"]["tones"][j].set_visible(True)
                        self.sweepData[file]["speed"]["tones"][j].set_visible(True)
                    else:
                        self.sweepData[file]["points"]["tones"][j].set_visible(False)
                        self.sweepData[file]["speed"]["tones"][j].set_visible(False)

                    if self.ui.actionMinimum_S21.isChecked():
                        self.sweepData[file]["points"]["minS21"][j].set_visible(True)
                        self.sweepData[file]["speed"]["minS21"][j].set_visible(True)
                    else:
                        self.sweepData[file]["points"]["minS21"][j].set_visible(False)
                        self.sweepData[file]["speed"]["minS21"][j].set_visible(False)

                    if self.ui.actionMax_Speed.isChecked():
                        self.sweepData[file]["points"]["maxSpeed"][j].set_visible(True)
                        self.sweepData[file]["speed"]["maxSpeed"][j].set_visible(True)
                        self.sweepData[file]["speed"]["smooth"][j][0].set_visible(True)
                    else:
                        self.sweepData[file]["points"]["maxSpeed"][j].set_visible(False)
                        self.sweepData[file]["speed"]["maxSpeed"][j].set_visible(False)
                        self.sweepData[file]["speed"]["smooth"][j][0].set_visible(False)

                    data_x = self.sweepData[file]["sweep"][j][0].get_xdata()
                    max_x = np.max(data_x)
                    min_x = np.min(data_x)

                    data_y = self.sweepData[file]["sweep"][j][0].get_ydata()
                    max_y = np.max(data_y)
                    min_y = np.min(data_y)

                    if max_x >= x_sweep_max:
                        x_sweep_max = max_x

                    if min_x <= x_sweep_min:
                        x_sweep_min = min_x

                    if max_y >= y_sweep_max:
                        y_sweep_max = max_y

                    if min_y <= y_sweep_min:
                        y_sweep_min = min_y

                    dx = data_x[1] - data_x[0]

                    data_x_sd = self.sweepData[file]["speed"]["signal"][j][0].get_xdata()
                    max_x_sd = np.max(data_x_sd)
                    min_x_sd = np.min(data_x_sd)

                    data_y_sd = self.sweepData[file]["speed"]["signal"][j][0].get_ydata()
                    max_y_sd = np.max(data_y_sd)
                    min_y_sd = np.min(data_y_sd)

                    if max_x_sd >= x_speed_max:
                        x_speed_max = max_x_sd

                    if min_x_sd <= x_speed_min:
                        x_speed_min = min_x_sd

                    if max_y_sd >= y_speed_max:
                        y_speed_max = max_y_sd

                    if min_y_sd <= y_speed_min:
                        y_speed_min = min_y_sd

                    dx_sd = data_x_sd[1] - data_x_sd[0]

                    rescale = True

                else:
                    self.sweepData[file]["sweep"][j][0].set_visible(False)
                    self.sweepData[file]["points"]["tones"][j].set_visible(False)
                    self.sweepData[file]["iq"][j][0].set_visible(False)
                    self.sweepData[file]["speed"]["signal"][j][0].set_visible(False)

                    self.sweepData[file]["points"]["tones"][j].set_visible(False)
                    self.sweepData[file]["speed"]["tones"][j].set_visible(False)

                    self.sweepData[file]["points"]["minS21"][j].set_visible(False)
                    self.sweepData[file]["speed"]["minS21"][j].set_visible(False)

                    self.sweepData[file]["points"]["maxSpeed"][j].set_visible(False)
                    self.sweepData[file]["speed"]["maxSpeed"][j].set_visible(False)
                    self.sweepData[file]["speed"]["smooth"][j][0].set_visible(False)

        if rescale:
            #if x_sweep_min == np.inf or x_sweep_max == -np.inf or y_sweep_min == np.inf or y_sweep_max == -np.inf or x_speed_min == np.inf or x_speed_max == -np.inf or y_speed_min == np.inf or y_speed_max == -np.inf: 
            try:
                self.fsweep.set_xlim([x_sweep_min - 5*dx, x_sweep_max + 5*dx])
                self.fsweep.set_ylim([y_sweep_min - 0.5, y_sweep_max + 0.5])

                self.fspeed.set_xlim([x_speed_min - 5*dx_sd, x_speed_max + 5*dx_sd])
                self.fspeed.set_ylim([y_speed_min, y_speed_max])
            except:
                pass

        self.figHomo_sweep.tight_layout()
        self.fsweep.figure.canvas.draw()

        self.figHomo_iq.tight_layout()
        self.fiq.figure.canvas.draw()

        self.figHomo_speed.tight_layout()
        self.fspeed.figure.canvas.draw()

    def plotSweepSelected(self, item, column):
        self.updateSweepPlots()

    def plotVNASelected(self, item, column):
        checked = dict()
        root = self.ui.plotsVNATree.invisibleRootItem()
        signal_count = root.childCount()
        rescale = False

        x_vna_max = -np.inf
        x_vna_min = np.inf

        y_vna_max = -np.inf
        y_vna_min = np.inf

        for i in range(signal_count):
            signal = root.child(i)
            checked_sweeps = list()
            num_children = signal.childCount()

            file = signal.text(0)
            for j in range(num_children):
                child = signal.child(j)
                if child.checkState(0) == QtCore.Qt.Checked:
                    checked_sweeps.append(child.text(0))
                    self.vnaData[file]["sweep"][j][0].set_visible(True)

                    data_x = self.vnaData[file]["sweep"][j][0].get_xdata()
                    max_x = np.max(data_x)
                    min_x = np.min(data_x)

                    data_y = self.vnaData[file]["sweep"][j][0].get_ydata()
                    max_y = np.max(data_y)
                    min_y = np.min(data_y)

                    if max_x >= x_vna_max:
                        x_vna_max = max_x

                    if min_x <= x_vna_min:
                        x_vna_min = min_x

                    if max_y >= y_vna_max:
                        y_vna_max = max_y

                    if min_y <= y_vna_min:
                        y_vna_min = min_y

                    dx = data_x[1] - data_x[0]

                    rescale = True
                else:
                    self.vnaData[file]["sweep"][j][0].set_visible(False)

        if rescale:
            self.fVNA.set_xlim([x_vna_min - 5*dx, x_vna_max + 5*dx])
            self.fVNA.set_ylim([y_vna_min - 0.5, y_vna_max + 0.5])

        self.figVNA.tight_layout()
        self.fVNA.figure.canvas.draw()

    def plotTimeStreamSelected(self, item, column):
        checked = dict()
        root = self.ui.timeTree.invisibleRootItem()
        signal_count = root.childCount()

        for i in range(signal_count):
            signal = root.child(i)
            checked_sweeps = list()
            num_children = signal.childCount()

            file = signal.text(0)
            for j in range(num_children):
                child = signal.child(j)
                if child.checkState(0) == QtCore.Qt.Checked:
                    checked_sweeps.append(child.text(0))
                    self.timeData[file]["time"][j][0].set_visible(True)
                    self.timeData[file]["psd"][j][0].set_visible(True)
                else:
                    self.timeData[file]["time"][j][0].set_visible(False)
                    self.timeData[file]["psd"][j][0].set_visible(False)

        self.figTimeStream.tight_layout()
        self.fts.figure.canvas.draw()

        self.figPSD_ts.tight_layout()
        self.fPSD.figure.canvas.draw()

    def updatePlotTree(self):
        self.clearTree(self.ui.plotsTree)
        tree = self.ui.plotsTree

        i = 0
        for key in self.sweepData.keys():
            parent = QTreeWidgetItem(tree)
            parent.setText(0, key)
            parent.setFlags(parent.flags() | Qt.ItemIsTristate | Qt.ItemIsUserCheckable)

            j = 0
            for j in range(len(self.sweepData[key]["sweep"])):
                child = QTreeWidgetItem(parent)
                #child.setIcon(0, QIcon('./resources/f0_tem.png'))
                child.setFlags(child.flags() | Qt.ItemIsTristate | Qt.ItemIsUserCheckable)
                child.setText(0, self.sweepData[key]["names"][j])
                child.setCheckState(0, Qt.Unchecked)
                j += 1
            i += 1

    def updatePlotVNATree(self):
        tree = self.ui.plotsVNATree
        self.clearTree(tree)

        i = 0
        for key in self.vnaData.keys():
            parent = QTreeWidgetItem(tree)
            parent.setText(0, key)
            parent.setFlags(parent.flags() | Qt.ItemIsTristate | Qt.ItemIsUserCheckable)

            j = 0
            for j in range(len(self.vnaData[key]["sweep"])):
                child = QTreeWidgetItem(parent)
                #child.setIcon(0, QIcon('./resources/f0_tem.png'))
                child.setFlags(child.flags() | Qt.ItemIsTristate | Qt.ItemIsUserCheckable)
                child.setText(0, self.vnaData[key]["names"][j])
                child.setCheckState(0, Qt.Unchecked)
                j += 1
            i += 1

    def updatePlotTimeTree(self):
        tree = self.ui.timeTree
        self.clearTree(tree)

        i = 0
        for key in self.timeData.keys():
            parent = QTreeWidgetItem(tree)
            parent.setText(0, key)
            parent.setFlags(parent.flags() | Qt.ItemIsTristate | Qt.ItemIsUserCheckable)

            j = 0
            for j in range(len(self.timeData[key]["time"])):
                child = QTreeWidgetItem(parent)
                #child.setIcon(0, QIcon('./resources/f0_tem.png'))
                child.setFlags(child.flags() | Qt.ItemIsTristate | Qt.ItemIsUserCheckable)
                child.setText(0, self.timeData[key]["names"][j])
                child.setCheckState(0, Qt.Unchecked)
                j += 1
            i += 1

    def newSweepPlot(self, event):
        w = QWidget()

        self.sweepData = {}
        try:
            self.figHomo_sweep.clf()
            self.figHomo_iq.clf()
            self.figHomo_speed.clf()
        except Exception as e:
            pass
        #try:
        file = self.ui.plotSweepEdit.text()
        self.sweepPlot(file)
        #except Exception as e:
        #    QMessageBox.information(w, "Error ploting", "Error plotting directory. " + str(e))

    def addSweepPlot(self, event):
        try:
            file = self.ui.plotSweepEdit.text()
            self.sweepPlot(file)
        except Exception as e:
            QMessageBox.information(w, "Error ploting", "Error adding a new plot. " + str(e))

    def newVNAPlot(self, event):
        w = QWidget()

        self.vnaData = {}
        try:
            self.figVNA.clf()
        except Exception as e:
            pass
        try:
            file = self.ui.plotVNAEdit.text()
            self.vnaPlot(file)
        except Exception as e:
            QMessageBox.information(w, "Error ploting", "Error plotting directory. " + str(e))

    def addVNAPlot(self, event):
        try:
            file = self.ui.plotVNAEdit.text()
            self.vnaPlot(file)
        except Exception as e:
            QMessageBox.information(w, "Error ploting", "Error adding a new plot. " + str(e))

    def newTimeStreamPlot(self, event):
        w = QWidget()

        self.timeData = {}
        try:
            self.figTimeStream.clf()
            self.figPSD_ts.clf()
        except Exception as e:
            pass
        try:
            file = self.ui.plotTimeStreamEdit.text()
            self.timePlot(file)
        except Exception as e:
            QMessageBox.information(w, "Error ploting", "Error plotting directory. " + str(e))

    def addTimeStreamPlot(self, event):
        w = QWidget()

        try:
            file = self.ui.plotTimeStreamEdit.text()
            self.timePlot(file)
        except Exception as e:
            QMessageBox.information(w, "Error ploting", "Error adding a new plot. " + str(e))

    def showSweepTones(self, event):
        self.updateSweepPlots()

    def maxSpeed(self, event):
        self.updateSweepPlots()

    def minS21(self, event):
        self.updateSweepPlots()

    def sweepPlot(self, file):
        
        self.sweepData[file] = {}

        roachID = self.ui.roach_ID.currentText()
        freqs, data = plot_sweep_avg.get_sweep_data(file)

        # Sweep
        self.fsweep = self.figHomo_sweep.add_subplot(111)
        # IQ Circle
        self.fiq = self.figHomo_iq.add_subplot(111)
        # Speed
        self.fspeed = self.figHomo_speed.add_subplot(111)

        channels = sorted(data.keys())
        auxSweep = []
        auxTones = []
        auxMinS21 = []
        auxMaxSpeed = []

        auxIQ = []

        auxSpeed = []
        auxSmooth = []
        auxTones_Speed = []
        auxMinS21_Speed = []
        auxMaxSpeed_Speed = []
        
        for chan in channels:
            # Magnitude
            mag = plot_sweep_avg.get_mag_log(data[chan])
            sweep = self.fsweep.plot(freqs[chan],mag,'o-')

            tn = np.median(freqs[chan])
            single_minS21 = freqs[chan][np.argmin(mag)]

            tones = self.fsweep.axvline(tn, color="#FF0033", linewidth=0.75)
            minS21 = self.fsweep.axvline(single_minS21, color="#3300FF", linewidth=0.75)
                        
            auxTones.append(tones)
            auxMinS21.append(minS21)
            auxSweep.append(sweep)
            
            # IQ Circles
            iq = self.fiq.plot(data[chan].real, data[chan].imag, 'p-')
            auxIQ.append(iq)
            
            # Speed
            freq, speed = plot_sweep_avg.get_speed(freqs[chan], data[chan])
            
            sm_speed = savgol_filter(speed, 7, 3)
            single_maxSpeed = freq[np.argmax(sm_speed)]

            speedPt = self.fspeed.plot(freq, speed, '*-')
            smooth_speedPt = self.fspeed.plot(freq, sm_speed, '--')

            tonesSpeed = self.fspeed.axvline(tn, color="#FF0033", linewidth=0.75)
            minS21Speed = self.fspeed.axvline(single_minS21, color="#3300FF", linewidth=0.75)
            maxSpeed = self.fsweep.axvline(single_maxSpeed, color="#33FF00", linewidth=0.75)
            maxSpeed_Speed = self.fspeed.axvline(single_maxSpeed, color="#33FF00", linewidth=0.75)

            auxTones_Speed.append(tonesSpeed)
            auxMinS21_Speed.append(minS21Speed)
            auxMaxSpeed_Speed.append(maxSpeed_Speed)
            auxMaxSpeed.append(maxSpeed)
            auxSpeed.append(speedPt)
            auxSmooth.append(smooth_speedPt)

            if not self.ui.actionShow_Tones.isChecked():
                tones.set_visible(False)
                tonesSpeed.set_visible(False)
            if not self.ui.actionMinimum_S21.isChecked():
                minS21.set_visible(False)
                minS21Speed.set_visible(False)
            if not self.ui.actionMax_Speed.isChecked():
                maxSpeed.set_visible(False)
                maxSpeed_Speed.set_visible(False)
                smooth_speedPt[0].set_visible(False)

        self.sweepData[file]["names"] = channels
        self.sweepData[file]["sweep"] = auxSweep
        self.sweepData[file]["points"] = {}
        self.sweepData[file]["points"]["tones"] = auxTones
        self.sweepData[file]["points"]["minS21"] = auxMinS21
        self.sweepData[file]["points"]["maxSpeed"] = auxMaxSpeed
        self.sweepData[file]["iq"] = auxIQ
        self.sweepData[file]["speed"] = {}
        self.sweepData[file]["speed"]["signal"] = auxSpeed
        self.sweepData[file]["speed"]["smooth"] = auxSmooth
        self.sweepData[file]["speed"]["tones"] = auxTones_Speed
        self.sweepData[file]["speed"]["minS21"] = auxMinS21_Speed
        self.sweepData[file]["speed"]["maxSpeed"] = auxMaxSpeed_Speed
        self.sweepData[file]["speed"]["smooth"] = auxSmooth

        self.fsweep.set_xlabel(r'$\textbf{frequency[Hz]}$')
        self.fsweep.set_ylabel(r"$\textbf{V [dB]}$")

        self.fsweep.grid(True)
        self.figHomo_sweep.tight_layout()
        self.fsweep.figure.canvas.draw()

        #self.fiq.set_xlabel(r'$\textbf{I}$')
        #self.fiq.set_ylabel(r'$\textbf{Q}$')

        self.fiq.grid(True)
        self.figHomo_iq.tight_layout()
        self.fiq.figure.canvas.draw()

        #self.fspeed.set_xlabel(r'$\textbf{frequency[Hz]}$')
        #self.fspeed.set_ylabel(r'$\textbf{Speed}$')

        self.fspeed.grid(True)
        self.figHomo_speed.tight_layout()
        self.fspeed.figure.canvas.draw()

        self.updatePlotTree()

    def vnaPlot(self, file):
        self.vnaData[file] = {}

        roachID = self.ui.roach_ID.currentText()

        # Ways to upload the toneslist
        freqs, data = plot_sweep_avg.get_sweep_data(file)

        # VNA Sweep
        self.fVNA = self.figVNA.add_subplot(111)

        channels = sorted(data.keys())
        auxVNA = []
        for chan in channels:
            # Magnitude
            mag = plot_sweep_avg.get_mag_log(data[chan])
            sweep = self.fVNA.plot(freqs[chan],mag,'o-')
            auxVNA.append(sweep)

        self.vnaData[file]["names"] = channels
        self.vnaData[file]["sweep"] = auxVNA

        self.fVNA.set_xlabel(r'$\textbf{frequency[Hz]}$')
        self.fVNA.set_ylabel(r'$\textbf{V[dB]}$')

        self.fVNA.grid(True)
        self.figVNA.tight_layout()
        self.fVNA.figure.canvas.draw()

        self.updatePlotVNATree()

    def timePlot(self, file):
        self.timeData[file] = {}

        roachID = self.ui.roach_ID.currentText()

        time, time_stream = plot_sweep_avg.get_stream_data(file)

        time = time-time[0]
        # Time Stream
        self.fts = self.figTimeStream.add_subplot(111)
        # PSD Time Stream
        self.fPSD = self.figPSD_ts.add_subplot(111)

        auxTime = []
        auxPSD = []

        double_channels = sorted(time_stream.keys())
        channels = [chan[:-2] for chan in double_channels if chan.endswith("_I")]

        for chan in channels:
            I = time_stream[chan + '_I']
            Q = time_stream[chan + '_Q']

            # Magnitude
            mag = plot_sweep_avg.get_mag_from_IQ(I,Q)
            tstream = self.fts.plot(time, mag)
            tstream[0].set_visible(False)

            #psd = np.abs(np.fft.fft(mag))

            freq, psd = signal.periodogram(mag, 488)
            psd = 10*np.log10(psd)
            #freq = np.linspace(0.0, 488.0/2.0, len(psd))
            psdStream = self.fPSD.semilogx(freq[1:], psd[1:])
            psdStream[0].set_visible(False)

            auxTime.append(tstream)
            auxPSD.append(psdStream)

        self.timeData[file]["names"] = channels
        self.timeData[file]["time"] = auxTime
        self.timeData[file]["psd"] = auxPSD

        self.fts.set_xlabel(r'$\textbf{Time[s]}$')
        self.fts.set_ylabel(r'$\textbf{V [n]}$')

        self.fts.grid(True)
        self.figTimeStream.tight_layout()
        self.fts.figure.canvas.draw()

        self.fPSD.set_xlabel(r'$\textbf{frequency[Hz]}$')
        self.fPSD.set_ylabel(r'$\textbf{PSD V[dB]}$')

        self.fPSD.grid(True)
        self.figPSD_ts.tight_layout()        
        self.fPSD.figure.canvas.draw()

        self.updatePlotTimeTree()
        
    # Sweep
    def addmpl_homodyne_sweep(self,fig):
        self.canvas_H = FigureCanvas(fig)
        self.ui.HomoPlot.addWidget(self.canvas_H)
        self.canvas_H.draw()
        self.toolbar_H = NavigationToolbar(self.canvas_H,
           self, coordinates=True)
        self.ui.HomoPlot.addWidget(self.toolbar_H)

    def rmmpl_homodyne_sweep(self):
        self.ui.HomoPlot.removeWidget(self.canvas_H)
        self.canvas_H.close()
        self.ui.HomoPlot.removeWidget(self.toolbar_H)
        self.toolbar_H.close()

    # IQ Circles
    def addmpl_homodyne_iq(self,fig):
        self.canvas_HIQ = FigureCanvas(fig)
        self.ui.IQPlot.addWidget(self.canvas_HIQ)
        self.canvas_HIQ.draw()
        self.toolbar_HIQ = NavigationToolbar(self.canvas_HIQ,
           self, coordinates=True)
        self.ui.IQPlot.addWidget(self.toolbar_HIQ)

    def rmmpl_homodyne_iq(self):
        self.ui.IQPlot.removeWidget(self.canvas_HIQ)
        self.canvas_HIQ.close()
        self.ui.IQPlot.removeWidget(self.toolbar_HIQ)
        self.toolbar_HIQ.close()

    # Speed
    def addmpl_homodyne_speed(self,fig):
        self.canvas_HSpeed = FigureCanvas(fig)
        self.ui.SpeedPlot.addWidget(self.canvas_HSpeed)
        self.canvas_HSpeed.draw()
        self.toolbar_HSpeed = NavigationToolbar(self.canvas_HSpeed,
           self, coordinates=True)
        self.ui.SpeedPlot.addWidget(self.toolbar_HSpeed)

    def rmmpl_homodyne_speed(self):
        self.ui.SpeedPlot.removeWidget(self.canvas_HSpeed)
        self.canvas_HSpeed.close()
        self.ui.SpeedPlot.removeWidget(self.toolbar_HSpeed)
        self.toolbar_HSpeed.close()

    def addmpl_vna(self,fig):
        self.canvas_V = FigureCanvas(fig)
        self.ui.VNAPlot.addWidget(self.canvas_V)
        self.canvas_V.draw()
        self.toolbar_V = NavigationToolbar(self.canvas_V,
           self, coordinates=True)
        self.ui.VNAPlot.addWidget(self.toolbar_V)

    def rmmpl_vna(self):
        self.ui.VNAPlot.removeWidget(self.canvas_V)
        self.canvas_V.close()
        self.ui.VNAPlot.removeWidget(self.toolbar_V)
        self.toolbar_V.close()

    def addmpl_ts(self,fig):
        self.canvas_ts = FigureCanvas(fig)
        self.ui.timeStreamPlot.addWidget(self.canvas_ts)
        self.canvas_ts.draw()
        self.toolbar_ts = NavigationToolbar(self.canvas_ts,
           self, coordinates=True)
        self.ui.timeStreamPlot.addWidget(self.toolbar_ts)

    def rmmpl_ts(self):
        self.ui.timeStreamPlot.removeWidget(self.canvas_ts)
        self.canvas_ts.close()
        self.ui.timeStreamPlot.removeWidget(self.toolbar_ts)
        self.toolbar_ts.close()

    def addmpl_ts_PSD(self,fig):
        self.canvas_ts_PSD = FigureCanvas(fig)
        self.ui.PSDPlot.addWidget(self.canvas_ts_PSD)
        self.canvas_ts_PSD.draw()
        self.toolbar_ts_PSD = NavigationToolbar(self.canvas_ts_PSD,
           self, coordinates=True)
        self.ui.PSDPlot.addWidget(self.toolbar_ts_PSD)

    def rmmpl_ts_PSD(self):
        self.ui.PSDPlot.removeWidget(self.canvas_ts_PSD)
        self.canvas_ts_PSD.close()
        self.ui.PSDPlot.removeWidget(self.toolbar_ts_PSD)
        self.toolbar_ts_PSD.close()

    # Threads Messages
    def msg_thread(self,text):
        logging.info(cm.INFO + text + cm.ENDC)

        # Write the LCD Value
        inA, outA = self.cal_att.readAtt()

        self.ui.inAttDisplay.display(inA)
        self.ui.OutAttDisplay.display(outA)

    def set_progressBar(self,value):
        self.ui.progressBar.setValue(int(100*value))

    def about(self):
        w = QWidget()
        w.resize(320, 240)
        w.setWindowTitle("Select directory to save the Tone List")
        QMessageBox.information(w, "About KID-ANALYSER", "KID Lab V1.0 by INAOE KID team, Marcial Becerril, Jose Miguel and Edgar Castillo. The core code is pcp developed by Pete Barry, Krusty Krab and Sam Rowe :), thanks a lot for the valuable help!")

    def filterTones(self, tones, overlap_dist=20e3):
        
        new_filter_tonelist = []
        new_tonelist = tones#np.sort(tones)

        for i in range(len(new_tonelist)-1):
            if (new_tonelist[i+1] - new_tonelist[i]) > overlap_dist:
                new_filter_tonelist.append(new_tonelist[i])

        # The last element
        if (new_tonelist[-1] - new_tonelist[-2]) > overlap_dist:
            new_filter_tonelist.append(new_tonelist[-1])

        return new_filter_tonelist

    def moveTones(self, event):
        checked = dict()
        header = "Name\tFreq\tOffset att\tAll\tNone\n"
        root = self.ui.plotsTree.invisibleRootItem()
        signal_count = root.childCount()

        for i in range(signal_count):
            signal = root.child(i)
            num_children = signal.childCount()

            roachID = self.ui.roach_ID.currentText()
            
            file = self.nRoach[roachID]["obj"].toneslist.tonelistfile
            file_nm = open(file,'w')
            file_nm.write(header)

            new_file = signal.text(0)

            maxSpeedTones = [tone.get_xdata()[0] for tone in self.sweepData[new_file]["points"]["maxSpeed"]]
            maxSpeedTones = self.filterTones(maxSpeedTones)

            minS21Tones = [tone.get_xdata()[0] for tone in self.sweepData[new_file]["points"]["minS21"]]
            minS21Tones = self.filterTones(minS21Tones)

            print maxSpeedTones

            for j in range(num_children):
                child = signal.child(j)
                tone = child.text(0)

                maxSpeed = int(maxSpeedTones[j])
                minS21 = int(minS21Tones[j])

                if self.ui.actionMax_Speed.isChecked():
                    row = tone + '\t' + str(maxSpeed) + "\t0\t1\t0\n"
                elif self.ui.actionMinimum_S21.isChecked():
                    row = tone + '\t' + str(minS21) + "\t0\t1\t0\n"

                file_nm.write(row)
            file_nm.close()
            
            self.nRoach[roachID]["obj"].toneslist.load_tonelist(file)
            self.updateTonesTable(roachID)

    def get_kid_params(self, file):
        freqs, data = plot_sweep_avg.get_sweep_data(file)

        channels = data.keys()

        didf_channels = []
        dqdf_channels = []
        I_f0 = []
        Q_f0 = []

        for chan in channels:
            # Get f0
            f0 = np.median(freqs[chan])

            # Get I and Q
            I = data[chan].real
            Q = data[chan].imag

            # Get the didf and dqdf
            didf = plot_sweep_avg.get_dxdf(freqs[chan], I, smooth=True, order=3, npoints=15)       
            dqdf = plot_sweep_avg.get_dxdf(freqs[chan], Q, smooth=True, order=3, npoints=15)    

            # Get df
            #df = plot_sweep_avg.get_df(I, Q, didf, dqdf, f0)

            didf_f0.append(didf[f0])
            dqdf_f0.append(dqdf[f0])

            I_f0.append(I[f0])
            Q_f0.append(Q[f0])

        return didf_f0, dqdf_f0, I_f0, Q_f0    

    def getSweepFiles(self):
        checked = dict()
        root = self.ui.plotsTree.invisibleRootItem()
        signal_count = root.childCount()

        files = []

        for i in range(signal_count):
            signal = root.child(i)
            checked_sweeps = list()
            num_children = signal.childCount()

            file = signal.text(0)
            files.append(file)

            print file

        """
        for file in files:
            didf_f0, dqdf_f0, I_f0, Q_f0 = self.get_kid_params(file)
            for chan in channels:
                # Writing Fits files
                hdr = self.writeFitsFile(file)

                hdr['IF0'] = I_f0
                hdr['QF0'] = Q_f0

                hdr['DQDF'] = didf_f0
                hdr['DIDF'] = dqdf_f0

                hdr['SAMPLERA'] = f0

                kid_number = hdr['TONE']
                input_att = hdr['INPUTATT']

                f0 = hdr['SYNTHFRE']
        """    

    def writeFitsFile(self):
        """
        Header
        """
        hdr = fits.Header()
        hdr['PROJECT'] = self.ui.projectEdit.text()
        hdr['DUT'] = self.ui.dutEdit.text()
        hdr['EXPERIMENT'] = self.ui.settingBox.text()
        hdr['CRYOSTAT'] = self.ui.cryostatBox.text()
        hdr['COLD_AMPLIFIER'] = self.ui.coldAmpInput.text()
        hdr['WARM_AMPLIFIER'] = self.ui.warmAmpOutput.text()
        hdr['IN_ATTEN'] = self.ui.inAttenRF.text()
        hdr['OUT_ATTEN'] = self.ui.outAttenRF.text()
        hdr['MODULATOR'] = self.ui.modRFEdit.text()
        hdr['DEMODULATOR'] = self.ui.demodRFEdit.text()
        hdr['SPLITTER'] = self.ui.splitterModelEdit.text()
        hdr['SYNTHESIZER'] = self.ui.synthModEdit.text()

        return hdr

    def exitKIDLab(self):
        for roach in self.nRoach.keys():
            self.nRoach[roach]["obj"].shutdown()
        self.ui.close()

app = QtGui.QApplication(sys.argv)
MyWindow = MainWindow()
sys.exit(app.exec_())
