import sys
import string
import os
import arcpy
import arcpy.sa
import time
import numpy as np
import shutil

arcpy.CheckOutExtension("Spatial")
arcpy.AddMessage(arcpy.GetParameterAsText(3))
arcpy.env.workspace = arcpy.GetParameterAsText(3)#r'in_memory'
#arcpy.AddMessage(arcpy.ListSpatialReferences())

def getAbsPath(layer):
	if layer:
		desc = arcpy.Describe(layer)
		path = desc.path
		return os.path.join(str(path),layer).replace(os.sep, '/')

squares = arcpy.GetParameterAsText(0)
in_ref = arcpy.Describe(squares).spatialReference
dsm = getAbsPath(arcpy.GetParameterAsText(1))
ort = getAbsPath(arcpy.GetParameterAsText(2))
wse_field = arcpy.GetParameterAsText(4)
chainage_field = arcpy.GetParameterAsText(5)
#arcpy.AddMessage(squares)
#arcpy.AddMessage(dsm)
#arcpy.AddMessage(ort)

def resetDir(dir):
	if os.path.exists(dir):
		shutil.rmtree(dir)
	os.makedirs(dir)
	return dir
tmp_dir = resetDir(os.path.join(arcpy.GetParameterAsText(3),"tmp"))
ort_dir = resetDir(os.path.join(arcpy.GetParameterAsText(3),"ort"))
dsm_dir = resetDir(os.path.join(arcpy.GetParameterAsText(3),"dsm"))

tmp_dir = tmp_dir.replace(os.sep, '/')
#tmp_raster = os.path.join(tmp_dir,"tmp_raster.tif").replace(os.sep, '/')
i=1
coord = arcpy.SpatialReference(r"Geographic Coordinate Systems/World/WGS 1984")
arcpy.AddMessage("coord_OK")
arcpy.AddMessage(arcpy.Describe(squares).dataType)
arcpy.FeatureClassToFeatureClass_conversion(squares,"/","squares.shp")
squares_sorted = arcpy.Sort_management("squares.shp","squares_sorted.shp",chainage_field)
arcpy.AddMessage(arcpy.Describe("squares_sorted.shp").dataType)
with arcpy.da.SearchCursor(squares_sorted ,['SHAPE@',wse_field, chainage_field], sql_clause=(None,'ORDER BY {}'.format(chainage_field))) as cursor:
    for row in cursor:
		arcpy.AddMessage(row[1])
		polygon = row[0]
		centroid = arcpy.PointGeometry(polygon.centroid, in_ref)
		arcpy.env.extent = polygon.extent
		#arcpy.MakeFeatureLayer_management(row[0], "mask_layer")
		#arcpy.AddMessage(row[0].dataType)
		#arcpy.Clip_analysis(in_features, clip_features, out_feature_class, {cluster_tolerance})
		#arcpy.sa.ExtractByMask(dsm, row[0])
		#arcpy.Clip_management(dsm, row[0], tmp_raster)
		#FC = arcpy.CreateFeatureclass_management("in_memory", "FC", "POLYGON", "", "DISABLED", "DISABLED", Coordinate_System, "", "0", "0", "0")
		arcpy.FeatureVerticesToPoints_management(polygon, "vertices.shp")
		v_dsc = arcpy.Describe("vertices.shp")
		with arcpy.da.SearchCursor("vertices.shp" ,'SHAPE@XY') as v_cursor:
			v_list = []
			for v_row in v_cursor:
				v_list.append(arcpy.Point(v_row[0][0],v_row[0][1]))
			#for point in points:
			#	arcpy.AddMessage(str(point))

			dsm_extract = arcpy.sa.ExtractByPolygon(dsm, v_list)
			ort_extract = arcpy.sa.ExtractByPolygon(ort, v_list)
			#dsm_extract.save(tmp_dir+"/"+str(row[1])+".tif")
			dsm_extract_arr = arcpy.RasterToNumPyArray(dsm_extract)
			ort_extract_arr = arcpy.RasterToNumPyArray(ort_extract)
			#ul = arcpy.Sort_management("vertices.shp", arcpy.Geometry(),[["SHAPE", "ASCENDING"]], "UL")[0]
			#lr = arcpy.Sort_management("vertices.shp", arcpy.Geometry(),[["SHAPE", "ASCENDING"]], "LR")[0]
			arcpy.Project_management(centroid,"centroid.shp",coord)
			#arcpy.Project_management(lr,"lr.shp",coord)
			centroid = arcpy.CopyFeatures_management("centroid.shp", arcpy.Geometry())[0].firstPoint
			#lr = arcpy.CopyFeatures_management("lr.shp", arcpy.Geometry())[0]
			arcpy.AddMessage(str(centroid.X)+" "+str(centroid.Y))
			#arcpy.AddMessage(str(lr.firstPoint.X)+" "+str(lr.firstPoint.Y))
			np.save(os.path.join(dsm_dir,"{}_{}_{}_{}_{}.npy".format(str(i).zfill(3),row[1],row[2],centroid.Y,centroid.X)),dsm_extract_arr)
			if ort_extract_arr.shape[0]==4:
				ort_extract_arr = ort_extract_arr[:3]
			np.save(os.path.join(ort_dir,"{}_{}_{}_{}_{}.npy".format(str(i).zfill(3),row[1],row[2],centroid.Y,centroid.X)),ort_extract_arr)
			arcpy.AddMessage(dsm_extract_arr.shape)
			arcpy.AddMessage(ort_extract_arr.shape)
			i+=1
		#arcpy.AddMessage(out)
		#arcpy.AddMessage(arcpy.Describe(dsm_part))
		#dsm_part.save(arcpy.GetParameterAsText(3))
		#dsm_part = arcpy.RasterToNumPyArray(dsm_part)
		#arcpy.AddMessage(dsm_part.shape)































