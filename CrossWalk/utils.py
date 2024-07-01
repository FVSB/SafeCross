from datetime import datetime
from functools import wraps

import numpy as np
from IPython.lib.display import Audio
from ultralytics import YOLO

# Crear el tono como una onda sinusoidal de NumPy
wave = np.sin(2 * np.pi * 500 * np.arange(15000 * 2) / 15000)


# Función para reproducir el tono
def play_sound():
    return Audio(wave, rate=10000, autoplay=True)


# Decorador para manejar excepciones
def handle_exceptions(func):
    """
    A decorator to handle exceptions in a function. If an exception occurs while executing the function,
    it plays a sound and prints an error message.

    Parameters:
    func (function): The function to be decorated.

    Returns:
    function: The decorated function.
    """

    @wraps(func)
    def wrapper(*args, **kwargs):
        """
        The wrapper function to execute the decorated function and handle exceptions.

        Parameters:
        *args: Variable length argument list of the decorated function.
        **kwargs: Arbitrary keyword arguments of the decorated function.

        Returns:
        The return value of the decorated function.
        """
        try:
            print(f"The function {func.__name__} was called at {datetime.now()}")
            return func(*args, **kwargs)
        except Exception as e:
            play_sound()
            print(f"An error occurred at {datetime.now()}: {e}")

    return wrapper


# Decorador para registrar el tiempo
def clock(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        start_time = datetime.now()
        print(f"La función {func.__name__} empezó a las {start_time}")

        result = func(*args, **kwargs)

        end_time = datetime.now()
        duration = end_time - start_time
        print(f"La función {func.__name__} terminó a las {end_time}")
        print(f"La duración total fue de {duration}")

        return result

    return wrapper


@clock
@handle_exceptions
def train(
        data_path: str,
        pre_train_model_path: str = '',
        save_model_name:str='modelo_fine_tuned.pt',
        epochs: int = 50,
        save_period: int = 1,
        imgsz: int = 512,
        batch: int = 32,
        patience=50,
        seed: int = 123,
        training_results=r'C:\Users\paco2\Documents\GitHub\MachineLearningProyect\CrossWalk\proyect',
        use_gpu: bool = True

        ):
    """
    Configurar los parámetros de entrenamiento
    epochs = 50  # Número de épocas de entrenamiento
    save_period = 1  # Guardar el modelo cada 10 épocas

    """
    # Cargar el modelo pre-entrenado
    model = YOLO('yolov8n.pt')  # Carga el modelo YOLOv8n pre-entrenado
    results = None
    if pre_train_model_path in [None, '', ' ']:
        # Realizar el fine-tuning
        results = model.train(
            data=data_path,
            val=True,
            epochs=epochs,
            imgsz=imgsz,
            save_period=save_period,
            device="0" if use_gpu else "cpu",  # Utilizar GPU 0
            project=training_results,
            batch=batch,
            cache=True,  # Almacena las imágenes del conjunto de datos en RAM
            verbose=True,
            seed=seed,
            patience=patience,
            resume=True,
            model=pre_train_model_path

        )
    else:
        results = model.train(
            data=data_path,
            val=True,
            epochs=epochs,
            imgsz=imgsz,
            save_period=save_period,
            device="0" if use_gpu else "cpu",  # Utilizar GPU 0
            project=training_results,
            batch=batch,
            cache=True,  # Almacena las imágenes del conjunto de datos en RAM
            verbose=True,
            seed=seed,
            patience=patience,
            resume=True,
        )


    # Guardar el modelo final
    model.save(save_model_name)

    return model,results
