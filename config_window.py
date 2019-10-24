#!/usr/bin/env python
# -*- coding: utf-8 -*-
# The homodyne system for AIG lab
#
# Copyright (C) November, 2018  Becerril, Marcial <mbecerrilt@inaoep.mx>
# Author: Becerril, Marcial <mbecerrilt@inaoep.mx> based in the codes of
# Pete Barry et al., Sam Gordon <sbgordo1@asu.edu>, Sam Rowe and Thomas Gascard.
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

import sys
import numpy as np
import yaml

from PyQt4 import QtCore, QtGui,uic
from PyQt4.QtCore import *
from PyQt4.QtGui import QPalette,QWidget,QFileDialog,QMessageBox, QTreeWidgetItem, QIcon, QPixmap

# Main Configuration Window
class configWindow(QtGui.QWidget):

    # Signal to connect with the main gui
    saveRoachSig = QtCore.pyqtSignal(dict)
    saveNetSig = QtCore.pyqtSignal(dict)
    saveHardSig = QtCore.pyqtSignal(dict)
    saveDirySig = QtCore.pyqtSignal(dict)
    saveLoggsSig = QtCore.pyqtSignal(dict)

    def __init__(self,parent, roach_config, network_config, filesys_config, hardware_config, logging_config):
        super(configWindow, self).__init__(parent)

        # Load of main window GUI
        # The GUI was developed in QT Designer
        self.ui = uic.loadUi("./src/gui/config.ui")

        # Screen dimensions
        screen = QtGui.QDesktopWidget().screenGeometry()

        self.size_x = screen.width()
        self.size_y = screen.height()

        self.ui.setWindowFlags(self.ui.windowFlags() | QtCore.Qt.CustomizeWindowHint)
        self.ui.setWindowFlags(self.ui.windowFlags() & ~QtCore.Qt.WindowMaximizeButtonHint)

        self.ui.move(self.size_x/2-self.ui.width()/2, self.size_y/2-self.ui.height()/2)

        self.hdw_cfg = hardware_config
        self.fsy_cfg = filesys_config
        self.lgg_cfg = logging_config
        self.net_cfg = network_config
        self.rch_cfg = roach_config

        # ROACH
        # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
        roach_list = self.rch_cfg.keys()
        network_list = self.net_cfg.keys()
        synth_list = self.hdw_cfg["synth_config"].keys()
        atten_list = self.hdw_cfg["atten_config"].keys()

        self.ui.nameRoachBox.clear()
        self.ui.netNameRoach.clear()
        self.ui.nameRoachBox.addItems(roach_list)
        self.ui.netNameRoach.addItems(roach_list)

        self.fillSynthesizer(synth_list)
        self.fillAttenuator(atten_list)

        # Roach settings
        self.updateRoach(roach_list[0])

        # Network settings
        self.updateNetwork(roach_list[0])

        # Hardware settings
        # Synthesizer
        self.ui.synthNameEdit.clear()
        self.ui.synthNameEdit.addItems(synth_list)
        self.updateSynth(synth_list[0])
        # Attenuator
        self.ui.attenNameEdit.clear()
        self.ui.attenNameEdit.addItems(atten_list)
        self.updateAtt(atten_list[0])

        # Directories
        self.updateDirectories()

        # Logging
        self.updateLogging()

        # Buttons
        # Roach
        self.ui.firmFileBtn.mousePressEvent = self.chooseFirmFilePath
        self.ui.tonesFileBtn.mousePressEvent = self.chooseTonesFilePath

        self.ui.nameRoachBox.currentIndexChanged.connect(self.changeRoach)

        self.ui.addRoachBtn.mousePressEvent = self.addNewRoach
        self.ui.removeRoachBtn.mousePressEvent = self.rmRoach

        self.ui.roachBtn.mousePressEvent = self.saveRoach
        self.ui.roachWriteBtn.mousePressEvent = self.writeRoach

        # Network
        self.ui.netNameRoach.currentIndexChanged.connect(self.changeRoachNetwork)

        self.ui.netBtn.mousePressEvent = self.saveNet
        self.ui.netWriteBtn.mousePressEvent = self.writeNet

        # Hardware
        self.ui.synthNameEdit.currentIndexChanged.connect(self.changeSynth)
        self.ui.attenNameEdit.currentIndexChanged.connect(self.changeAtten)

        self.ui.activeSynthBtn.mousePressEvent = self.toggleActiveSynth
        self.ui.activeAttenBtn.mousePressEvent = self.toggleActiveAtten

        self.ui.addSynthBtn.mousePressEvent = self.addNewSynth
        self.ui.removeSynthBtn.mousePressEvent = self.rmSynth

        self.ui.addAttenBtn.mousePressEvent = self.addNewAtten
        self.ui.removeAttenBtn.mousePressEvent = self.rmAtten

        self.ui.hdwBtn.mousePressEvent = self.saveHardware
        self.ui.hdwWriteBtn.mousePressEvent = self.writeHardware

        # Directories
        self.ui.diryFirmwareEdit.mousePressEvent = self.chooseFirmPath
        self.ui.rootDiryBtn.mousePressEvent = self.chooseRootPath
        self.ui.logDiryBtn.mousePressEvent = self.chooseLogPath
        self.ui.pidBtn.mousePressEvent = self.choosePIDPath
        self.ui.toneDiryBtn.mousePressEvent = self.chooseTonesPath
        self.ui.RAMDiryBtn.mousePressEvent = self.chooseRAMPath

        self.ui.theDiryBtn.mousePressEvent = self.saveDiry
        self.ui.theDiryWriteBtn.mousePressEvent = self.writeDiry

        # TODO. Load number of roaches and their parameters

    def fillSynthesizer(self, synth):
        self.ui.synthLOModel.clear()
        self.ui.synthCLKModel.clear()

        # Synthesizer
        lo_list = ["None"]
        clk_list = ["None"]
        for dev in synth:
            if self.hdw_cfg["synth_config"][dev]["active"]:
                if self.hdw_cfg["synth_config"][dev]["type"] == "lo":
                    lo_list.append(dev)
                elif self.hdw_cfg["synth_config"][dev]["type"] == "clock":
                    clk_list.append(dev)

        self.ui.synthLOModel.addItems(lo_list)
        self.ui.synthCLKModel.addItems(clk_list)

    def fillAttenuator(self, atten):
        self.ui.modelInputBox.clear()
        self.ui.modelOutputBox.clear()

        # Attenuator
        input_list = ["None"]
        output_list = ["None"]
        for dev in atten:
            if self.hdw_cfg["atten_config"][dev]["active"]:
                if self.hdw_cfg["atten_config"][dev]["direction"] == "in":
                    input_list.append(dev)
                elif self.hdw_cfg["atten_config"][dev]["direction"] == "out":
                    output_list.append(dev)

        self.ui.modelInputBox.addItems(input_list)
        self.ui.modelOutputBox.addItems(output_list)

    def updateRoach(self, roachID):
        # ROACH Name
        self.ui.nameRoachEdit.setText(roachID)

        # Files
        self.ui.firmfileEdit.setText(self.rch_cfg[roachID]["firmware_file"])
        self.ui.toneListFileEdit.setText(self.rch_cfg[roachID]["tonelist_file"])

        # Synthesizers
        # CLK
        synthCLKitems = [self.ui.synthCLKModel.itemText(i) for i in range(self.ui.synthCLKModel.count())]
        try:
            indCLK = synthCLKitems.index(self.rch_cfg[roachID]["synthid_clk"])
        except ValueError:
            indCLK = 0
        self.ui.synthCLKModel.setCurrentIndex(indCLK)

        # CLK
        synthLOitems = [self.ui.synthLOModel.itemText(i) for i in range(self.ui.synthLOModel.count())]
        try:
            indLO = synthLOitems.index(self.rch_cfg[roachID]["synthid_lo"])
        except ValueError:
            indLO = 0

        self.ui.synthLOModel.setCurrentIndex(indLO)

        self.ui.stepSynthEdit.setText(str(self.rch_cfg[roachID]["sweep_step"]))
        self.ui.spanSynthEdit.setText(str(self.rch_cfg[roachID]["sweep_span"]))
        self.ui.avgsSynthEdit.setText(str(self.rch_cfg[roachID]["sweep_avgs"]))

        # Parameters
        self.ui.maxQueueEdit.setText(str(self.rch_cfg[roachID]["max_queue_len"]))
        self.ui.dacBwEdit.setText(str(self.rch_cfg[roachID]["dac_bandwidth"]))
        self.ui.roachAccumEdit.setText(str(self.rch_cfg[roachID]["roach_accum_len"]))
        self.ui.ddsEdit.setText(str(self.rch_cfg[roachID]["dds_shift"]))
        self.ui.bufferLenEdit.setText(str(self.rch_cfg[roachID]["buffer_len_to_write"]))

        # Attenuators
        # Input
        attenInputitems = [self.ui.modelInputBox.itemText(i) for i in range(self.ui.modelInputBox.count())]
        try:
            indInput = attenInputitems.index(self.rch_cfg[roachID]["att_in"])
        except ValueError:
            indInput = 0
        self.ui.modelInputBox.setCurrentIndex(indInput)

        # Output
        attenOutputitems = [self.ui.modelOutputBox.itemText(i) for i in range(self.ui.modelOutputBox.count())]
        try:
            indOutput = attenOutputitems.index(self.rch_cfg[roachID]["att_out"])
        except ValueError:
            indOutput = 0
        self.ui.modelOutputBox.setCurrentIndex(indOutput)

        # Test comb
        self.ui.centerFreqEdit.setText(str(self.rch_cfg[roachID]["center_freq"]))
        self.ui.nFreqsEdit.setText(str(self.rch_cfg[roachID]["Nfreq"]))
        self.ui.offsetFreqEdit.setText(str(self.rch_cfg[roachID]["symm_offset"]))
        self.ui.maxPosFreqEdit.setText(str(self.rch_cfg[roachID]["max_pos_freq"]))
        self.ui.minPosFreqEdit.setText(str(self.rch_cfg[roachID]["min_pos_freq"]))
        self.ui.maxNegFreqEdit.setText(str(self.rch_cfg[roachID]["max_neg_freq"]))
        self.ui.minNegFreqEdit.setText(str(self.rch_cfg[roachID]["min_neg_freq"]))

    def clearRoachFields(self):
        # ROACH Name
        self.ui.nameRoachEdit.clear()

        """
        # Files
        self.ui.firmfileEdit.clear()
        self.ui.toneListFileEdit.clear()

        # Synthesizers
        # CLK
        indCLK = 0
        self.ui.synthCLKModel.setCurrentIndex(indCLK)

        # LO
        indLO = 0
        self.ui.synthLOModel.setCurrentIndex(indLO)

        self.ui.stepSynthEdit.clear()
        self.ui.spanSynthEdit.clear()
        self.ui.avgsSynthEdit.clear()

        # Parameters
        self.ui.maxQueueEdit.clear()
        self.ui.dacBwEdit.clear()
        self.ui.roachAccumEdit.clear()
        self.ui.ddsEdit.clear()
        self.ui.bufferLenEdit.clear()

        # Attenuators
        # Input
        indInput = 0
        self.ui.modelInputBox.setCurrentIndex(indInput)

        # Output
        indOutput = 0
        self.ui.modelOutputBox.setCurrentIndex(indOutput)

        # Test comb
        self.ui.centerFreqEdit.clear()
        self.ui.nFreqsEdit.clear()
        self.ui.offsetFreqEdit.clear()
        self.ui.maxPosFreqEdit.clear()
        self.ui.minPosFreqEdit.clear()
        self.ui.maxNegFreqEdit.clear()
        self.ui.minNegFreqEdit.clear()
        """

    def clearSynthFields(self):
        self.ui.nameSynthEdit.clear()

        self.ui.typeSynthBox.setCurrentIndex(0)
        self.ui.vendorSynthBox.setCurrentIndex(0)

        self.ui.modelSynthEdit.clear()
        self.ui.serialSynthEdit.clear()
        self.ui.channelSynthEdit.clear()

    def clearAttenFields(self):
        self.ui.nameAttenEdit.clear()

        self.ui.directionAttenBox.setCurrentIndex(0)
        self.ui.vendorAttenBox.setCurrentIndex(0)
        self.ui.attenLinkedEdit.clear()

        self.ui.modelAttenEdit.clear()
        self.ui.serialAttenEdit.clear()
        self.ui.channelAttenEdit.clear()

    def updateNetwork(self, roachID):

        self.ui.roachppcEdit.setText(self.net_cfg[roachID]["roach_ppc_ip"])
        self.ui.socketEdit.setText(str(self.net_cfg[roachID]["socket_type"]))
        self.ui.deviceIDEdit.setText(str(self.net_cfg[roachID]["device_id"]))

        # UDP Dest
        self.ui.udpDestDevEdit.setText(self.net_cfg[roachID]["udp_dest_device"])
        self.ui.udpDestIPEdit.setText(self.net_cfg[roachID]["udp_dest_ip"])
        self.ui.udpDestMACEdit.setText(str(self.net_cfg[roachID]["udp_dest_mac"]))
        self.ui.udpDestPortEdit.setText(str(self.net_cfg[roachID]["udp_dest_port"]))

        # UDP Source
        self.ui.udpSourceIPEdit.setText(self.net_cfg[roachID]["udp_source_ip"])
        self.ui.udpSourceMACEdit.setText(str(self.net_cfg[roachID]["udp_source_mac"]))
        self.ui.udpSourcePortEdit.setText(str(self.net_cfg[roachID]["udp_source_port"]))

        self.ui.bufferSizeEdit.setText(str(self.net_cfg[roachID]["buf_size"]))
        self.ui.headerLenEdit.setText(str(self.net_cfg[roachID]["header_len"]))

    def updateSynth(self, synthID):
        self.ui.nameSynthEdit.setText(str(synthID))

        if self.hdw_cfg["synth_config"][synthID]["active"]:
            self.ui.activeSynthBtn.setStyleSheet("""QWidget {
                                        color: white;
                                        background-color: green
                                        }""")
        else:
            self.ui.activeSynthBtn.setStyleSheet("""QWidget {
                                        color: white;
                                        background-color: red
                                        }""")

        if self.hdw_cfg["synth_config"][synthID]["type"] == "lo":
            self.ui.typeSynthBox.setCurrentIndex(1)
        elif self.hdw_cfg["synth_config"][synthID]["type"] == "clock":
            self.ui.typeSynthBox.setCurrentIndex(0)

        try:
            vendor = self.hdw_cfg["synth_config"][synthID]["vendor"].lower()
            vendorItems = [self.ui.vendorSynthBox.itemText(i).lower() for i in range(self.ui.vendorSynthBox.count())]
            indVendor = vendorItems.index(vendor)
        except ValueError:
            indVendor = 0

        self.ui.vendorSynthBox.setCurrentIndex(indVendor)

        self.ui.modelSynthEdit.setText(self.hdw_cfg["synth_config"][synthID]["model"])
        self.ui.serialSynthEdit.setText(str(self.hdw_cfg["synth_config"][synthID]["serial"]))
        self.ui.channelSynthEdit.setText(str(self.hdw_cfg["synth_config"][synthID]["channel"]))

    def updateAtt(self, attenID):
        self.ui.nameAttenEdit.setText(str(attenID))

        if self.hdw_cfg["atten_config"][attenID]["active"]:
            self.ui.activeAttenBtn.setStyleSheet("""QWidget {
                                        color: white;
                                        background-color: green
                                        }""")
        else:
            self.ui.activeAttenBtn.setStyleSheet("""QWidget {
                                        color: white;
                                        background-color: red
                                        }""")
        if self.hdw_cfg["atten_config"][attenID]["direction"] == "in":
            self.ui.directionAttenBox.setCurrentIndex(1)
        elif self.hdw_cfg["atten_config"][attenID]["direction"] == "out":
            self.ui.directionAttenBox.setCurrentIndex(0)

        self.ui.attenLinkedEdit.setText(self.hdw_cfg["atten_config"][attenID]["linked"])

        vendor = self.hdw_cfg["atten_config"][attenID]["vendor"].lower()
        vendorItems = [self.ui.vendorAttenBox.itemText(i).lower() for i in range(self.ui.vendorAttenBox.count())]
        try:
            indVendor = vendorItems.index(vendor)
        except ValueError:
            indVendor = 0
        self.ui.vendorAttenBox.setCurrentIndex(indVendor)

        self.ui.modelAttenEdit.setText(self.hdw_cfg["atten_config"][attenID]["model"])
        self.ui.serialAttenEdit.setText(str(self.hdw_cfg["atten_config"][attenID]["serial"]))
        self.ui.channelAttenEdit.setText(str(self.hdw_cfg["atten_config"][attenID]["channel"]))

    def updateDirectories(self):
        self.ui.rootDiryEdit.setText(self.fsy_cfg["rootdir"])
        self.ui.lggFilesEdit.setText(str(self.fsy_cfg["logfiledir"]))
        self.ui.diryPIDEdit.setText(str(self.fsy_cfg["pidfiledir"]))
        self.ui.dirySaveEdit.setText(self.fsy_cfg["savedatadir"])
        self.ui.diryToneListEdit.setText(str(self.fsy_cfg["tonelistdir"]))
        self.ui.diryRamDiskEdit.setText(str(self.fsy_cfg["livefiledir"]))
        self.ui.diryFirmwareEdit.setText(str(self.fsy_cfg["firmwaredir"]))

    def updateLogging(self):
        self.ui.lggFileEdit.setText(str(self.lgg_cfg["logfilename"]))
        self.ui.lgghostEdit.setText(str(self.lgg_cfg["serverconfig"]["host"]))
        self.ui.lggPortEdit.setText(str(self.lgg_cfg["serverconfig"]["port"]))

        if self.lgg_cfg["disable_existing_loggers"]:
            self.ui.activeLoggsBtn.setStyleSheet("""QWidget {
                                        color: white;
                                        background-color: green
                                        }""")
        else:
            self.ui.activeLoggsBtn.setStyleSheet("""QWidget {
                                        color: white;
                                        background-color: red
                                        }""")

    def showGUI(self):
        self.ui.show()

    def updateNumRoach(self, event):
        self.nRoach = self.ui.numRoachEdit.value()
        self.ui.numRoachNwEdit.setValue(self.nRoach)
        self.numRoach(self.nRoach)

    def updateNumNwRoach(self, event):
        self.nRoach = self.ui.numRoachNwEdit.value()
        self.ui.numRoachEdit.setValue(self.nRoach)
        self.numRoach(self.nRoach)

    def saveRoach(self, event):
        currentName = self.ui.nameRoachBox.currentText()

        # Firmware
        self.rch_cfg[currentName]["firmware_file"] = self.ui.firmfileEdit.text()
        self.rch_cfg[currentName]["tonelist_file"] = self.ui.toneListFileEdit.text()

        # Synthesizers
        self.rch_cfg[currentName]["sweep_step"] = self.ui.stepSynthEdit.text()
        self.rch_cfg[currentName]["sweep_span"] = self.ui.spanSynthEdit.text()
        self.rch_cfg[currentName]["sweep_avgs"] = self.ui.avgsSynthEdit.text()

        self.rch_cfg[currentName]["synthid_lo"] = self.ui.synthLOModel.currentText()
        self.rch_cfg[currentName]["synthid_clk"] = self.ui.synthCLKModel.currentText()

        # Parameters
        self.rch_cfg[currentName]["max_queue_len"] = self.ui.maxQueueEdit.text()
        self.rch_cfg[currentName]["dac_bandwidth"] = self.ui.dacBwEdit.text()
        self.rch_cfg[currentName]["roach_accum_len"] = self.ui.roachAccumEdit.text()
        self.rch_cfg[currentName]["dds_shift"] = self.ui.ddsEdit.text()
        self.rch_cfg[currentName]["buffer_len_to_write"] = self.ui.bufferLenEdit.text()

        # Attenuators
        self.rch_cfg[currentName]["att_in"] = self.ui.modelInputBox.currentText()
        self.rch_cfg[currentName]["att_out"] = self.ui.modelOutputBox.currentText()

        # Test Comb
        self.rch_cfg[currentName]["center_freq"] = self.ui.centerFreqEdit.text()
        self.rch_cfg[currentName]["Nfreq"] = self.ui.nFreqsEdit.text()
        self.rch_cfg[currentName]["symm_offset"] = self.ui.offsetFreqEdit.text()

        self.rch_cfg[currentName]["max_pos_freq"] = self.ui.maxPosFreqEdit.text()
        self.rch_cfg[currentName]["min_pos_freq"] = self.ui.minPosFreqEdit.text()
        self.rch_cfg[currentName]["max_neg_freq"] = self.ui.maxNegFreqEdit.text()
        self.rch_cfg[currentName]["min_neg_freq"] = self.ui.minNegFreqEdit.text()

        newName = self.ui.nameRoachEdit.text()

        self.rch_cfg[newName] = self.rch_cfg.pop(currentName)
        self.net_cfg[newName] = self.net_cfg.pop(currentName)

        updateList = self.rch_cfg.keys()

        self.ui.nameRoachBox.clear()
        self.ui.nameRoachBox.addItems(updateList)
        self.ui.nameRoachBox.setCurrentIndex(updateList.index(newName))

        self.ui.netNameRoach.clear()
        self.ui.netNameRoach.addItems(updateList)
        self.ui.netNameRoach.setCurrentIndex(updateList.index(newName))

        self.saveRoachSig.emit(self.rch_cfg)

    def writeRoach(self,event):
        self.saveRoach(event)
        with open('roach_config.cfg', 'w') as outfile:
            yaml.dump(self.rch_cfg, outfile, default_flow_style=False)

    def logFile(self, event):
        # Logging
        self.lgg_cfg["logfilename"] = self.ui.lggFileEdit.text()
        self.lgg_cfg["formatters"]["simple"]["format"] = self.ui.lggformatEdit.text()

        self.lgg_cfg["logrotatetime"] = self.ui.lggRotEdit.text()
        self.lgg_cfg["serverconfig"]["host"] = self.ui.lgghostEdit.text()
        self.lgg_cfg["serverconfig"]["port"] = self.ui.lggPortEdit.text()

        with open('logging_config.cfg', 'w') as outfile:
            yaml.dump(self.lgg_cfg, outfile, default_flow_style=False)

    def saveHardware(self, event):
        self.saveSynth()
        self.saveAtten()

        synth_list = self.hdw_cfg["synth_config"].keys()
        atten_list = self.hdw_cfg["atten_config"].keys()

        self.fillSynthesizer(synth_list)
        self.fillAttenuator(atten_list)

        self.saveHardSig.emit(self.hdw_cfg)

    def writeHardware(self, event):
        self.saveSynth()
        self.saveAtten()

        with open('hardware_config.cfg', 'w') as outfile:
            yaml.dump(self.hdw_cfg, outfile, default_flow_style=False)

    def saveSynth(self):
        currentName = self.ui.synthNameEdit.currentText()

        self.hdw_cfg["synth_config"][currentName]["type"] = self.ui.typeSynthBox.currentText().lower()
        self.hdw_cfg["synth_config"][currentName]["vendor"] = self.ui.vendorSynthBox.currentText()
        self.hdw_cfg["synth_config"][currentName]["model"] = self.ui.modelSynthEdit.text()
        self.hdw_cfg["synth_config"][currentName]["serial"] = self.ui.serialSynthEdit.text()
        self.hdw_cfg["synth_config"][currentName]["channel"] = self.ui.channelSynthEdit.text()

        newName = self.ui.nameSynthEdit.text()

        self.hdw_cfg["synth_config"][newName] = self.hdw_cfg["synth_config"].pop(currentName)
        updateList = self.hdw_cfg["synth_config"].keys()

        self.ui.synthNameEdit.clear()
        self.ui.synthNameEdit.addItems(updateList)
        self.ui.synthNameEdit.setCurrentIndex(updateList.index(newName))

    def saveAtten(self):
        currentName = self.ui.attenNameEdit.currentText()

        self.hdw_cfg["atten_config"][currentName]["direction"] = self.ui.directionAttenBox.currentText().lower()
        self.hdw_cfg["atten_config"][currentName]["linked"] = self.ui.attenLinkedEdit.text()

        self.hdw_cfg["atten_config"][currentName]["vendor"] = self.ui.vendorAttenBox.currentText()
        self.hdw_cfg["atten_config"][currentName]["model"] = self.ui.modelAttenEdit.text()
        self.hdw_cfg["atten_config"][currentName]["serial"] = self.ui.serialAttenEdit.text()
        self.hdw_cfg["atten_config"][currentName]["channel"] = self.ui.channelAttenEdit.text()

        newName = self.ui.nameAttenEdit.text()

        self.hdw_cfg["atten_config"][newName] = self.hdw_cfg["atten_config"].pop(currentName)
        updateItems = self.hdw_cfg["atten_config"].keys()

        self.ui.attenNameEdit.clear()
        self.ui.attenNameEdit.addItems(updateItems)
        self.ui.attenNameEdit.setCurrentIndex(updateItems.index(newName))

    def saveNet(self, event):
        netID = self.ui.netNameRoach.currentText()

        # Network
        self.net_cfg[netID]["roach_ppc_ip"] = self.ui.roachppcEdit.text()
        self.net_cfg[netID]["socket_type"] = self.ui.socketEdit.text()
        self.net_cfg[netID]["device_id"] = self.ui.deviceIDEdit.text()

        self.net_cfg[netID]["udp_dest_device"] = self.ui.udpDestDevEdit.text()
        self.net_cfg[netID]["udp_dest_ip"] = self.ui.udpDestIPEdit.text()
        self.net_cfg[netID]["udp_dest_mac"] = self.ui.udpDestMACEdit.text()
        self.net_cfg[netID]["udp_dest_port"] = self.ui.udpDestPortEdit.text()

        self.net_cfg[netID]["udp_source_ip"] = self.ui.udpSourceIPEdit.text()
        self.net_cfg[netID]["udp_source_mac"] = self.ui.udpSourceMACEdit.text()
        self.net_cfg[netID]["udp_source_port"] = self.ui.udpSourcePortEdit.text()

        self.net_cfg[netID]["buf_size"] = self.ui.bufferSizeEdit.text()
        self.net_cfg[netID]["header_len"] = self.ui.headerLenEdit.text()

        self.saveNetSig.emit(self.net_cfg)

    def writeNet(self,event):
        self.saveNet(event)

        with open('network_config.cfg', 'w') as outfile:
            yaml.dump(self.net_cfg, outfile, default_flow_style=False)

    def saveDiry(self, event):
        # Directory
        self.fsy_cfg["rootdir"] = self.ui.rootDiryEdit.text()
        self.fsy_cfg["logfiledir"] = self.ui.lggFilesEdit.text()
        self.fsy_cfg["pidfiledir"] = self.ui.diryPIDEdit.text()
        self.fsy_cfg["savedatadir"] = self.ui.dirySaveEdit.text()
        self.fsy_cfg["tonelistdir"] = self.ui.diryToneListEdit.text()
        self.fsy_cfg["livefiledir"] = self.ui.diryRamDiskEdit.text()
        self.fsy_cfg["firmwaredir"] = self.ui.diryFirmwareEdit.text()

        self.saveDirySig.emit(sefl.fsy_cfg)

    def writeDiry(self, event):
        self.saveDiry(event)

        with open('filesys_config.cfg', 'w') as outfile:
            yaml.dump(self.fsy_cfg, outfile, default_flow_style=False)

    def choosePath(self,flag):
        w = QWidget()
        w.resize(320, 240)
        w.setWindowTitle("Select")

        if flag == "firm":
            firmwareDiry = QFileDialog.getExistingDirectory(self, "Select Folder")
            self.ui.diryFirmwareEdit.setText(firmwareDiry)
        elif flag == "firmFile":
            firmwareFile = QFileDialog.getOpenFileName(self, "Select File")
            self.ui.firmfileEdit.setText(firmwareFile)
        elif flag == "root":
            root = QFileDialog.getExistingDirectory(self, "Select Folder")
            self.ui.rootDiryEdit.setText(root)
        elif flag == "logg":
            logg = QFileDialog.getExistingDirectory(self, "Select Folder")
            self.ui.lggFilesEdit.setText(logg)
        elif flag == "pid":
            pid = QFileDialog.getExistingDirectory(self, "Select Folder")
            self.ui.diryPIDEdit.setText(pid)
        elif flag == "tonelist":
            self.tones = QFileDialog.getExistingDirectory(self, "Select Folder")
            self.ui.diryToneListEdit.setText(self.tones)
        elif flag == "tonesFile":
            TonesFile = QFileDialog.getOpenFileName(self, "Select File")
            self.ui.toneListFileEdit.setText(TonesFile)
        elif flag == "ram":
            self.ram = QFileDialog.getExistingDirectory(self, "Select Folder")
            self.ui.diryRamDiskEdit.setText(self.ram)

    def chooseFirmPath(self,event):
        self.choosePath("firm")

    def chooseRootPath(self,event):
        self.choosePath("root")

    def chooseLogPath(self,event):
        self.choosePath("logg")

    def choosePIDPath(self,event):
        self.choosePath("pid")

    def chooseTonesPath(self,event):
        self.choosePath("tonelist")

    def chooseRAMPath(self,event):
        self.choosePath("ram")

    def chooseTonesFilePath(self,event):
        self.choosePath("tonesFile")

    def chooseFirmFilePath(self,event):
        self.choosePath("firmFile")

    def changeRoach(self,index):
        new_roach_selected = self.ui.nameRoachBox.currentText()
        try:
            self.updateRoach(new_roach_selected)
        except KeyError:
            pass

    def changeRoachNetwork(self,index):
        new_roach_selected = self.ui.netNameRoach.currentText()
        try:
            self.updateNetwork(new_roach_selected)
        except KeyError:
            pass

    def changeSynth(self,index):
        new_roach_selected = self.ui.synthNameEdit.currentText()
        try:
            self.updateSynth(new_roach_selected)
        except KeyError:
            pass

    def changeAtten(self,index):
        new_roach_selected = self.ui.attenNameEdit.currentText()
        try:
            self.updateAtt(new_roach_selected)
        except KeyError:
            pass

    def addNewRoach(self,event):
        self.rch_cfg["new_roach"] = self.emptyTemplateRoach()
        self.net_cfg["new_roach"] = self.emptyTemplateNetwork()
        self.ui.nameRoachBox.addItem("new_roach")
        self.ui.netNameRoach.addItem("new_roach")

        self.ui.nameRoachBox.setCurrentIndex(self.ui.nameRoachBox.count()-1)

        self.clearRoachFields()

    def addNewSynth(self,event):
        self.hdw_cfg["synth_config"]["new_synth"] = self.emptyTemplateSynth()
        self.ui.synthNameEdit.addItem("new_synth")
        self.ui.synthNameEdit.setCurrentIndex(self.ui.synthNameEdit.count()-1)

        self.clearSynthFields()

    def addNewAtten(self,event):
        self.hdw_cfg["atten_config"]["new_atten"] = self.emptyTemplateAtten()
        self.ui.attenNameEdit.addItem("new_atten")
        self.ui.attenNameEdit.setCurrentIndex(self.ui.attenNameEdit.count()-1)

        self.clearAttenFields()

    def rmRoach(self,event):
        currentSize = self.rch_cfg.keys()

        if len(currentSize) > 0:
            rmKey = self.ui.nameRoachBox.currentText()
            index = self.ui.nameRoachBox.currentIndex()
            del self.rch_cfg[rmKey]

            updateList = self.rch_cfg.keys()
            self.ui.nameRoachBox.clear()
            self.ui.nameRoachBox.addItems(updateList)

            if index > 0:
                index = index - 1
                self.updateSynth(updateList[index])

    def rmSynth(self,event):
        currentSize = self.hdw_cfg["synth_config"].keys()

        if len(currentSize) > 0:
            rmKey = self.ui.synthNameEdit.currentText()
            index = self.ui.synthNameEdit.currentIndex()
            del self.hdw_cfg["synth_config"][rmKey]

            updateList = self.hdw_cfg["synth_config"].keys()
            self.ui.synthNameEdit.clear()
            self.ui.synthNameEdit.addItems(updateList)

            if index > 0:
                index = index - 1
                self.updateSynth(updateList[index])

    def rmAtten(self,event):
        currentSize = self.hdw_cfg["atten_config"].keys()

        if len(currentSize) > 0:
            rmKey = self.ui.attenNameEdit.currentText()
            index = self.ui.attenNameEdit.currentIndex()
            del self.hdw_cfg["atten_config"][rmKey]

            updateList = self.hdw_cfg["atten_config"].keys()
            self.ui.attenNameEdit.clear()
            self.ui.attenNameEdit.addItems(updateList)

            if index > 0:
                index = index - 1
                self.updateAtt(updateList[index])

    def emptyTemplateRoach(self):
        roach = {
            "firmware_file" : None,
            "tonelist_file"   : None,
            "synthid_lo" : None,
            "synthid_clk": None, # null value allowed in this case of using external clock source
            "att_in"   : None,
            "att_out"   : None,
            "max_queue_len"   : 0, # maximum queue length from configuration file (1000 ~ 8 MB)
            "sweep_step": 0,
            "sweep_span": 0,
            "sweep_avgs": 0, # number of packets to average per LO step in a sweep
            "dac_bandwidth"   : 0,
            "roach_accum_len" : 0, # 2**roach_accum_len - 1
            "dds_shift": 0,
            "center_freq": 0,   # LO center frequency, MHz
            "Nfreq": 0,   # Number of frequencies in test com
            "max_pos_freq": 0,  # Maximum positive frequency, Hz
            "min_pos_freq": 0, # Minimum positive frequency, Hz
            "max_neg_freq": 0,   # Maximum negative frequency, Hz
            "min_neg_freq": 0,  # Minimum negative frequency, Hz
            "symm_offset": 0, # Offset between positive and negative combs, Hz
            "buffer_len_to_write": 0
        }

        return roach

    def emptyTemplateNetwork(self):
        network = {
            "device_id": None,
            "socket_type": None,
            "roach_ppc_ip": None,
            "udp_source_ip": None,
            "udp_source_port": None,
            "udp_source_mac": None,
            "udp_dest_ip": "localhost",
            "udp_dest_port": None,
            "udp_dest_device": None,
            "udp_dest_mac": None,
            "buf_size": 0,
            "header_len": 0
        }

        return network

    def emptyTemplateSynth(self):
        synth = {
            "active": False,
            "type": "clock",
            "vendor": "",
            "model": "",
            "serial": "",
            "channel": None,  # arguement for a channel index for multi-output synths - ignored for single channel synths
            "synthport": "",
            "comm_port": ""
        }

        return synth

    def emptyTemplateAtten(self):
        atten = {
            "active": False,
            "vendor": "",
            "model": "",
            "serial": "",
            "channel": None,
            "direction": "in",
            "linked": None
        }

        return atten

    def toggleActiveSynth(self,event):
        synthID = self.ui.synthNameEdit.currentText()

        if self.hdw_cfg["synth_config"][synthID]["active"]:
            self.ui.activeSynthBtn.setStyleSheet("""QWidget {
                                        color: white;
                                        background-color: red
                                        }""")
            self.hdw_cfg["synth_config"][synthID]["active"] = False
        else:
            self.ui.activeSynthBtn.setStyleSheet("""QWidget {
                                        color: white;
                                        background-color: green
                                        }""")
            self.hdw_cfg["synth_config"][synthID]["active"] = True

    def toggleActiveAtten(self,event):

        attenID = self.ui.attenNameEdit.currentText()

        if self.hdw_cfg["atten_config"][attenID]["active"]:
            self.ui.activeAttenBtn.setStyleSheet("""QWidget {
                                        color: white;
                                        background-color: red
                                        }""")
            self.hdw_cfg["atten_config"][attenID]["active"] = False
        else:
            self.ui.activeAttenBtn.setStyleSheet("""QWidget {
                                        color: white;
                                        background-color: green
                                        }""")
            self.hdw_cfg["atten_config"][attenID]["active"] = True
