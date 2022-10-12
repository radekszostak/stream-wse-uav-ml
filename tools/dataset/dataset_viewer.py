from PyQt5.QtWidgets import (QApplication, QComboBox, QGridLayout, QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton, QToolTip, QGroupBox)
from PyQt5.QtGui import (QPixmap, QImage, QFont)
import os
import csv
import numpy
import matplotlib.cm
from qimage2ndarray import array2qimage

CMAP = matplotlib.cm.get_cmap('viridis')

class MainWindow(QWidget):
    def __init__(self):
        super().__init__()
        os.chdir(os.getcwd())
        self.loadCsvData() #fills self.dataDictList
        self.names = [d['name'] for d in self.dataDictList]

        self.setWindowTitle("DSM Water Level - dataset viewer")
        self.outerLayout = QVBoxLayout()
        self.topLayout = QHBoxLayout(); self.outerLayout.addLayout(self.topLayout)
        self.xOrtPixmapLabel = QLabel(); #self.topLayout.addWidget(self.xOrtPixmapLabel)
        #self.xOrtPixmapLabel.setScaledContents(True)
        self.xOrtPixmapLabel.setMouseTracking(True)
        self.xOrtPixmapLabel.mouseMoveEvent = self.getPixel
        self.xOrtLayout = QVBoxLayout(); self.topLayout.addLayout(self.xOrtLayout)
        self.xOrtLayout.addWidget(QLabel('Ortophoto:'))
        self.xOrtLayout.addWidget(self.xOrtPixmapLabel)
        self.xDsmPixmapLabel = QLabel()
        #self.xDsmPixmapLabel.setScaledContents(True)
        self.xDsmPixmapLabel.setMouseTracking(True)
        self.xDsmPixmapLabel.mouseMoveEvent = self.getPixel
        self.xDsmLayout = QVBoxLayout(); self.topLayout.addLayout(self.xDsmLayout)
        self.xDsmLayout.addWidget(QLabel('Digital Surface Model:'))
        self.xDsmLayout.addWidget(self.xDsmPixmapLabel)
        #self.yDsmPixmapLabel = QLabel()
        #self.yDsmPixmapLabel.setScaledContents(True)
        #self.yDsmPixmapLabel.setMouseTracking(True)
        #self.yDsmPixmapLabel.mouseMoveEvent = self.getPixel
        #self.yDsmLayout = QVBoxLayout(); self.topLayout.addLayout(self.yDsmLayout)
        #self.yDsmLayout.addWidget(QLabel('y_dsm (DSM with denoised water surface):'))
        #self.yDsmLayout.addWidget(self.yDsmPixmapLabel)
        self.bottomLayout = QHBoxLayout(); self.outerLayout.addLayout(self.bottomLayout)
        
        self.gridLayout = QGridLayout()
        boldFont=QFont()
        boldFont.setBold(True)
        self.levelLabel = QLabel()
        #self.levelLabel.setFont(boldFont)
        self.meanLabel = QLabel(); #self.bottomLayout.addWidget(self.meanLabel)
        self.stdLabel = QLabel(); #self.bottomLayout.addWidget(self.stdLabel)
        self.minLabel = QLabel(); #self.bottomLayout.addWidget(self.minLabel)
        self.maxLabel = QLabel(); #self.bottomLayout.addWidget(self.maxLabel)
        self.latLabel = QLabel()
        self.lonLabel = QLabel()  
        self.chainLabel = QLabel()
        self.phaseLabel = QLabel()
        #self.levelLabelLabel=QLabel('Water level:')
        #self.levelLabelLabel.setFont(boldFont)
        self.gridLayout.addWidget(QLabel('Water level:'),0,0)
        self.gridLayout.addWidget(self.levelLabel,0,1)
        self.gridLayout.addWidget(QLabel('Chainage:'),0,2)
        self.gridLayout.addWidget(self.chainLabel,0,3)
        self.gridLayout.addWidget(QLabel('DSM mean:'),1,0)
        self.gridLayout.addWidget(self.meanLabel,1,1)
        self.gridLayout.addWidget(QLabel('DSM std:'),1,2)
        self.gridLayout.addWidget(self.stdLabel,1,3)
        self.gridLayout.addWidget(QLabel('DSM min:'),2,0)
        self.gridLayout.addWidget(self.minLabel,2,1)
        self.gridLayout.addWidget(QLabel('DSM max:'),2,2)
        self.gridLayout.addWidget(self.maxLabel,2,3)
        self.gridLayout.addWidget(QLabel('Latitude:'),3,0)
        self.gridLayout.addWidget(self.latLabel,3,1)
        self.gridLayout.addWidget(QLabel('Longitude:'),4,0)
        self.gridLayout.addWidget(self.lonLabel,4,1)
        self.gridLayout.addWidget(QLabel('Phase:'),3,2)
        self.gridLayout.addWidget(self.phaseLabel,3,3)
        self.buttonLayout = QGridLayout()
        self.bottomLayout.addLayout(self.gridLayout)
        self.nameBox = QComboBox(); self.buttonLayout.addWidget(self.nameBox,0,0,1,2)
        self.nameBox.activated.connect(self.nameBoxAction)
        self.prevBtn = QPushButton("<"); self.buttonLayout.addWidget(self.prevBtn,1,0)
        self.prevBtn.clicked.connect(self.prevBtnAction)
        self.nextBtn = QPushButton(">"); self.buttonLayout.addWidget(self.nextBtn,1,1)
        self.nextBtn.clicked.connect(self.nextBtnAction)
        self.bottomLayout.addLayout(self.buttonLayout)
        self.setLayout(self.outerLayout)
        self.nameBox.addItems(self.names)
        self.itemCount = len(self.names)
        self.currentItem = 0
        self.loadSample()
        

    def nextBtnAction(self):
        self.currentItem += 1
        self.loadSample()
    def prevBtnAction(self):
        self.currentItem -= 1
        self.loadSample()
    def nameBoxAction(self):
        self.currentItem = self.nameBox.currentIndex()
        self.loadSample()

    def loadCsvData(self):
        self.dataDictList = []
        with open('dataset.csv', newline='') as csvFile:
            csvReader = csv.reader(csvFile, delimiter=',')
            firstRow = True
            for row in csvReader:
                if firstRow:
                    firstRow = False
                    continue
                data_dict = {   'name': row[0],
                                'level': float(row[1]),
                                'mean': float(row[2]),
                                'std': float(row[3]),
                                'min': float(row[4]),
                                'max': float(row[5]),
                                'chain': float(row[6]),
                                'lat': float(row[7]),
                                'lon': float(row[8]),
                                'phase': row[9]
                            }
                self.dataDictList.append(data_dict)
        self.dataDictList = sorted(self.dataDictList, key=lambda x: x['name'])

    def loadSample(self):
        self.levelLabel.setText(str(round(self.dataDictList[self.currentItem]['level'],3)))
        self.meanLabel.setText(str(round(self.dataDictList[self.currentItem]['mean'],3)))
        self.stdLabel.setText(str(round(self.dataDictList[self.currentItem]['std'],3)))
        self.minLabel.setText(str(round(self.dataDictList[self.currentItem]['min'],3)))
        self.maxLabel.setText(str(round(self.dataDictList[self.currentItem]['max'],3)))
        self.chainLabel.setText(str(round(self.dataDictList[self.currentItem]['chain'],3)))
        self.latLabel.setText(str(round(self.dataDictList[self.currentItem]['lat'],5)))
        self.lonLabel.setText(str(round(self.dataDictList[self.currentItem]['lon'],5)))
        self.phaseLabel.setText(self.dataDictList[self.currentItem]['phase'])
        self.nameBox.setCurrentIndex(self.currentItem)
        if self.currentItem==0:
            self.prevBtn.setDisabled(1)
        else:
            self.prevBtn.setDisabled(0)
        if self.currentItem==self.itemCount-1:
            self.nextBtn.setDisabled(1)
        else:
            self.nextBtn.setDisabled(0)
        fileName = self.dataDictList[self.currentItem]['name']+".npy"
        self.xOrtNpy = numpy.moveaxis(numpy.load(os.path.join("ort",fileName)), 0, -1)
        self.xDsmNpy = numpy.load(os.path.join("dsm",fileName))

        self.imgSize = self.xDsmNpy.shape[0]
        minVal = self.dataDictList[self.currentItem]['min']
        maxVal = self.dataDictList[self.currentItem]['max']
        xDsmNpyRGB = (numpy.delete(CMAP((self.xDsmNpy-minVal)/(maxVal-minVal)),3,2)*255).astype(int)
        xOrtPixmap = QPixmap.fromImage(array2qimage(self.xOrtNpy))
        xDsmPixmap = QPixmap.fromImage(array2qimage(xDsmNpyRGB))
        #QImage(self.xOrtNpy, self.xOrtNpy.shape[1], self.xOrtNpy.shape[2], QImage.Format_RGB888))
        self.xOrtPixmapLabel.setPixmap(xOrtPixmap)
        self.xDsmPixmapLabel.setPixmap(xDsmPixmap)

    def getPixel(self, event):
        x = event.x()
        y = event.y()
        gPos = event.globalPos()
        gPos.setX(gPos.x()+10)
        gPos.setY(gPos.y()+10)
        if 0 <= x < self.imgSize and 0 <= y < self.imgSize:
            if self.xDsmPixmapLabel.underMouse():
                QToolTip.showText(gPos, str(self.xDsmNpy[y,x])+" m.a.s.l")
            elif self.xOrtPixmapLabel.underMouse():
                QToolTip.showText(gPos, str(self.xDsmNpy[y,x])+" m.a.s.l")
                #QToolTip.showText(gPos, f"R: {self.xOrtNpy[y,x,0]}, G: {self.xOrtNpy[y,x,1]}, B: {self.xOrtNpy[y,x,2]}")
        
if __name__ == '__main__':

    import sys
    try:
        app = QApplication(sys.argv)
        mainWindow = MainWindow()
        mainWindow.show()
        mainWindow.setFixedSize(mainWindow.size())
        sys.exit(app.exec_())
        
    except FileNotFoundError as err:
        print(err)
        print("\nMake sure that the dataset_viewer.exe application file is located in the dataset folder (that means along with csv files and dsm/ort directories).\n")
        input("Press Enter to continue...")
    except Exception as err:
        print(err)
        print("\nUnexpected error. Please contact the dataset author.\n")
        input("Press Enter to continue...")