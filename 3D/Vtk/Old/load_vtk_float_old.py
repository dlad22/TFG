import numpy as np

def load_vtk_float(filename):
    """
    Load VTK dataset.
    Usage: img, dim, spacing, origin = load_vtk_float(filename)
    
    Parameters:
    filename: str
        Path to the VTK file.
        
    Returns:
    img: np.ndarray
        The image data.
    dim: tuple
        Dimensions of the image.
    spacing: tuple
        Spacing of the image.
    origin: tuple
        Origin of the image.
    """
    try:
        with open(filename, 'rb') as f:
            # Skip the first few lines of the header
            f.readline()  # # vtk DataFile Version x.x
            f.readline()  # Comments
            f.readline()  # BINARY
            f.readline()  # DATASET STRUCTURED_POINTS
            
            # Read dimensions
            header = f.readline().decode('utf-8')  # DIMENSIONS NX NY NZ
            dim = tuple(map(int, header.split()[1:]))
            
            # Read spacing
            header = f.readline().decode('utf-8')  # SPACING SX SY SZ
            spacing = tuple(map(float, header.split()[1:]))
            
            # Read origin
            header = f.readline().decode('utf-8')  # ORIGIN OX OY OZ
            origin = tuple(map(float, header.split()[1:]))
            
            f.readline()  # POINT_DATA NXNYNZ
            f.readline()  # SCALARS float
            f.readline()  # LOOKUP_TABLE default

            # Mostrar f
            print(f"File: {filename}")
            print(f"Dimensions: {dim}")
            print(f"Spacing: {spacing}")
            print(f"Origin: {origin}")
        
            
            # Read the image data
            img = np.fromfile(f, dtype=np.float32, count=np.prod(dim))
            # buffer = f.read()  # Leer todo el contenido del archivo en un búfer de bytes

            # # Interpretar el búfer como un array de tipo float32
            # img = np.frombuffer(buffer, dtype=np.float32)

            print(img.size)
            print(img[0])


            # Contar numero de valores distintos de cero
            print(np.count_nonzero(img))
            

            # Verificar si hay algún valor nan en img
            # if np.isnan(img).any():
            #     print("Hay valores NaN en el array img.")
                
            #     # Mostrar los índices de los valores nan
            #     print(np.argwhere(np.isnan(img)))
            #     print(np.argwhere(np.isnan(img)).size)

            #     # Sustituir los valores nan por 0
            #     img = np.nan_to_num(img)
            #     print("Se han sustituido los valores NaN por 0.")


            # Calcular max y min de img
            max_img = np.max(img)
            min_img = np.min(img)
            print(f"Max: {max_img}, Min: {min_img}")
            
            # Reshape the image data
            img = img.reshape(dim)

            # Calcular max y min de img
            max_img = np.max(img)
            min_img = np.min(img)
            print(f"Max: {max_img}, Min: {min_img}")

            print(img.shape)
            print('FIN VTK')

            
    except FileNotFoundError:
        print(f"Error: El archivo '{filename}' no se pudo encontrar.")
    except Exception as e:
        print(f"Error al procesar el archivo '{filename}': {str(e)}")
            
    return img, dim, spacing, origin
