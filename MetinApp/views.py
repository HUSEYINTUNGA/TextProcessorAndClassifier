from django.shortcuts import render
import string
import re
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer, WordNetLemmatizer
from django.http import JsonResponse, HttpResponse
from train_model import predict_class
import pandas as pd
import os
from django.conf import settings
from datetime import datetime
import time

user_csv_file = None
selected_columns = None

def is_valid_word(word):
    """
    Kelimenin geçerli bir kelime olup olmadığını kontrol eder.
    Geçerli bir kelime, sadece harflerden oluşmalı ve uzunluğu 1'den büyük olmalıdır.
    """
    return len(word) > 1 and word.isalpha()

def clean_text(user_text, options):
    """
    Parametre olarak gelen metni temizler ve belirli işlemleri uygular.
    Uygulanabilecek işlemler:
    - Noktalama işaretlerini kaldırma.
    - Özel karakterleri kaldırma.
    - Harfleri büyük/küçük hale dönüştürme.
    - Durdurma kelimelerini (stopwords) kaldırma.
    - Stemming ve lemmatization işlemleri.
    """
    print("clean_text fonksiyonu çalıştı")
    print("clean_text fonksiyonu {} verisi için {} işlemlerini uygulayacak".format(user_text, options.values()))
    try:
        if options.get('remove_punctuation'):
            user_text = user_text.translate(str.maketrans('', '', string.punctuation))

        if options.get('remove_special_chars'):
            user_text = re.sub(r'[#$@{}\[\]\/\\)<>(|!\'^+%&/½=*&€~¨´æ£éß]', '', user_text)

        if options.get('convert_to_lowercase'):
            user_text = user_text.lower()

        if options.get('convert_to_uppercase'):
            user_text = user_text.upper()

        if options.get('remove_stopwords'):
            stop_words = set(stopwords.words('english'))
            user_text = ' '.join([word for word in user_text.split() if word.lower() not in stop_words])

        if options.get('stemming'):
            ps = PorterStemmer()
            user_text = ' '.join([ps.stem(word) for word in user_text.split() if is_valid_word(word)])

        if options.get('lemmatization'):
            lemmatizer = WordNetLemmatizer()
            user_text = ' '.join([lemmatizer.lemmatize(word) for word in user_text.split() if is_valid_word(word)])
        
        return user_text
    except Exception as e:
        return f"Text cleaning error: {str(e)}"

def HomePage(request):
    """
    Ana sayfa işlemlerini yöneten fonksiyon.
    Kullanıcıdan metin alır, belirli temizleme işlemleri ve sınıflandırma uygular.
    İşlenmiş metni ve sınıflandırma sonucunu JSON formatında döndürür.
    """
    if request.method == 'POST':
        user_text = request.POST.get('text', '')

        if not user_text.strip():
            return JsonResponse({'error': 'Lütfen metin girin.'})

        options = {
            'remove_punctuation': request.POST.get('remove_punctuation') == 'on',
            'remove_special_chars': request.POST.get('remove_special_chars') == 'on',
            'convert_to_lowercase': request.POST.get('convert_to_lowercase') == 'on',
            'convert_to_uppercase': request.POST.get('convert_to_uppercase') == 'on',
            'remove_stopwords': request.POST.get('remove_stopwords') == 'on',
            'stemming': request.POST.get('stemming') == 'on',
            'lemmatization': request.POST.get('lemmatization') == 'on',
        }

        if not any(options.values()) and not request.POST.get('classify_text'):
            return JsonResponse({'error': 'Lütfen en az bir işlem türü veya sınıflandırma seçin.'})

        processed_text = clean_text(user_text, options)
        classification_result = None

        if request.POST.get('classify_text'):
            try:
                classification_result = predict_class(user_text)
            except Exception as e:
                return JsonResponse({'error': f"Sınıf tahmini sırasında bir hata oluştu: {str(e)}"})

        return JsonResponse({
            'processed_text': processed_text,
            'classification_result': classification_result
        })

    return render(request, 'Home.html')

def UploadPage(request):
    """
    Kullanıcının bir CSV dosyası yüklemesine olanak tanır.
    Yüklenen CSV dosyasının sütun isimlerini alır ve ekranda gösterir.
    """
    print("UploadPage fonksiyonu çalıştı")
    global user_csv_file
    if request.method == 'POST' and request.FILES.get('csv_file'):
        csv_file = request.FILES['csv_file']

        if not csv_file.name.endswith('.csv'):
            return JsonResponse({'error': 'Lütfen bir CSV dosyası yükleyin.'}, status=400)

        try:
            user_csv_file = pd.read_csv(csv_file)
            columns = user_csv_file.columns.tolist()
            return render(request, 'Upload.html', {'columns': columns, 'message': 'CSV dosyası başarıyla yüklendi.'})
        except Exception as e:
            return JsonResponse({'error': f'Hata oluştu: {str(e)}'}, status=400)

    return render(request, 'Upload.html')

def process_columns(request):
    """
    Kullanıcının yüklenen CSV dosyasından sütunları seçmesine olanak tanır.
    Seçilen sütunlar global bir değişkene kaydedilir.
    """
    global user_csv_file
    global selected_columns
    if request.method == 'POST':
        selected_columns = request.POST.getlist('selected_columns')

        if not selected_columns:
            return JsonResponse({'error': 'Hiçbir sütun seçilmedi.'}, status=400)
        for column in selected_columns:
            print("Seçilen sütun/sütunlar : {}".format(column))

        return render(request, 'Upload.html', {'selected_columns': selected_columns, 'columns': user_csv_file.columns.tolist()})

    return render(request, 'Upload.html')

def process_text(request):
    """
    Seçilen sütunlardaki verileri temizler ve belirlenen işlemleri uygular.
    İşlenmiş veriler yeni bir CSV dosyasına kaydedilir ve kullanıcıya indirme bağlantısı sağlanır.
    """
    global selected_columns
    global user_csv_file

    if request.method == 'POST':
        if user_csv_file is None:
            return JsonResponse({'error': 'CSV dosyası yüklenmemiş.'}, status=400)
        print(selected_columns)    
        operations = {
            'remove_punctuation': request.POST.get('remove_punctuation') == 'on',
            'remove_special_chars': request.POST.get('remove_special_chars') == 'on',
            'convert_to_lowercase': request.POST.get('convert_to_lowercase') == 'on',
            'convert_to_uppercase': request.POST.get('convert_to_uppercase') == 'on',
            'remove_stopwords': request.POST.get('remove_stopwords') == 'on',
            'stemming': request.POST.get('stemming') == 'on',
            'lemmatization': request.POST.get('lemmatization') == 'on',
        }
        if not any(operations.values()):
            return JsonResponse({'error': 'Lütfen en az bir işlem seçin.'}, status=400)
            
        try:
            for column in selected_columns:
                if column not in user_csv_file.columns:
                    return JsonResponse({'error': f"'{column}' sütunu bulunamadı."}, status=400)
                for index in range(len(user_csv_file[column])):
                    original_value = user_csv_file[column].iloc[index]
                    processed_value = clean_text(original_value, operations)
                    user_csv_file.at[index, column] = processed_value
                    
            time.sleep(10)        
                    
            timestamp = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
            output_filename = f'processedCsv_{timestamp}.csv'
            output_path = os.path.join('MetinApp/UserCsvFiles', output_filename)

            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            user_csv_file.to_csv(output_path, index=False)

            with open(output_path, 'rb') as f:
                response = HttpResponse(f.read(), content_type='text/csv')
                response['Content-Disposition'] = f'attachment; filename="{output_filename}"'
                return response

        except Exception as e:
            return JsonResponse({'error': f'Hata oluştu: {str(e)}'}, status=400)

    return render(request, 'Upload.html')

def AboutMe(request):
    """
    Hakkımda sayfasını render eder.
    Bu sayfa, uygulama ve geliştirici hakkında bilgi içermektedir.
    """
    return render(request, 'AboutMe.html')
