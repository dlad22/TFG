import vtk
from vtk.util.numpy_support import vtk_to_numpy

def load_vtk_short(filename):
    # Crear un lector para archivos .vtk (datos estructurados)
    reader = vtk.vtkImageReader2()
    reader.SetFileName(filename)
    
    # Especificar el tipo de dato, int16 en este caso
    reader.SetDataScalarTypeToShort()
    reader.SetFileDimensionality(3)  # Especificar que es un volumen 3D
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

