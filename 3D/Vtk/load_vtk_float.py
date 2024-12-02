import vtk
from vtk.util.numpy_support import vtk_to_numpy

def load_vtk_float(filename):
    # Crear un lector para archivos .vtk (datos estructurados)
    reader = vtk.vtkStructuredPointsReader()
    reader.SetFileName(filename)
    reader.Update()

    # Obtener la imagen
    imageData = reader.GetOutput()

    # Obtener las dimensiones
    dim = imageData.GetDimensions()

    # Obtener el espaciado
    spacing = imageData.GetSpacing()

    # Obtener el origen
    origin = imageData.GetOrigin()

    # Leer los datos del punto
    pointData = imageData.GetPointData().GetScalars()
    img = vtk_to_numpy(pointData)
    
    # Remodelar la imagen según las dimensiones
    img = img.reshape(dim, order='F')
    
    return img, dim, spacing, origin

