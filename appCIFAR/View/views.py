from django.shortcuts import render, redirect
from appCIFAR.Logica import modelos
from django.templatetags.static import static
from django.conf import settings
from rest_framework.decorators import api_view
from django.http import JsonResponse

class Clasificacion:
    @staticmethod
    def determinarAprobacion(request):
        return render(request, "prediccionimagenes.html")

    @api_view(['POST'])
    def predecir(request):
        try:
            imagen = request.FILES.get('imagen')
            ruta_modelo_cnn = 'D:/P63/Aprendizaje Automattico/Unidad 2/ProyectoCIFAR/Recursos/CIFAR100model4CNN.h5'
            ruta_modelo_svm = 'D:/P63/Aprendizaje Automattico/Unidad 2/ProyectoCIFAR/Recursos/CIFAR100classififierSVM.pickle'

            modelo_cifar = modelos.ModeloCIFAR(ruta_modelo_cnn, ruta_modelo_svm)

            with open('temp_image.png', 'wb') as temp_image:
                for chunk in imagen.chunks():
                    temp_image.write(chunk)

            print("Realizando predicción...")
            resultado = modelo_cifar.predecirImagen('temp_image.png')
            print(f"Resultado de la predicción: {resultado}")
            resultado['image'] = static('temp_image.png')  # Asegúrate de ajustar esto según la estructura de tus rutas
            return render(request, "prediccionimagenes.html", {"resultado": resultado})

            # Redirige a la plantilla informeprediccion.html
          

        except Exception as e:
            print(f'Error inesperado en la vista predecir: {str(e)}')
            resultado = f'Ocurrió un error inesperado: {str(e)}'
            return render(request, "prediccionimagenes.html", {"e": resultado}, status=500)

    @api_view(['POST'])
    def predecir_io_json(request):
        try:
            ruta_imagen = request.data.get("imagen")

            modelo_cifar = modelos.ModeloCIFAR()
            
            print("Realizando predicción desde JSON...")
            resultado = modelo_cifar.predecirImagen(ruta_imagen)
            print(f"Resultado de la predicción desde JSON: {resultado}")

            # Redirige a la plantilla informeprediccion.html
            return render(request, "informeprediccion.html", {"resultado": resultado})

        except ValueError as e:
            data = {'error': f'Datos inválidos: {str(e)}'}
            return JsonResponse(data, status=400)
        except Exception as e:
            data = {'error': f'Ocurrió un error inesperado: {str(e)}'}
            return JsonResponse(data, status=500)
