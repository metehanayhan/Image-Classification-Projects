# Convolutional Neural Networks (CNN) ve Random Forest

Bu proje, **Temel Öğrenme Algoritmaları** dersi kapsamında hazırlanmış olup, Convolutional Neural Networks (CNN) ve Random Forest algoritmalarının karşılaştırılmasını içermektedir. Görsel veri üzerinde sınıflandırma yapmak amacıyla kullanılan bu algoritmalar, çeşitli metrikler üzerinden değerlendirilmiştir.

## Projenin Amacı
Bu proje, görüntü verileri üzerinde derin öğrenme yöntemleri ile geleneksel makine öğrenimi algoritmalarının karşılaştırmalı analizini yapmayı amaçlamaktadır. Özellikle, **kanser sınıflandırma** problemi üzerine odaklanılmıştır.

## Kullanılan Yöntemler ve Algoritmalar

### 1. **Convolutional Neural Networks (CNN)**
CNN, görsel verilerden anlamlı özellikleri çıkarmak için kullanılan bir derin öğrenme modelidir. Bu modelde:
- Görüntü işleme için **Convolutional**, **MaxPooling**, **Flatten** ve **Dense** katmanları kullanılmıştır.
- Çıkışta **Softmax aktivasyon fonksiyonu** ile sınıflandırma yapılmıştır.

#### CNN Yapısının Özeti:
- Conv2D (32 Filtre) -> MaxPooling -> Conv2D (64 Filtre) -> MaxPooling -> Flatten -> Dense (128 Nöron) -> Dense (Softmax, 2 Çıkış)

### 2. **Random Forest**
Random Forest, karar ağaçlarından oluşan bir topluluk öğrenme yöntemidir. Görsel verilerdeki özellikler önce klasik makine öğrenimi teknikleriyle işlenmiş ve ardından sınıflandırma yapılmıştır.

## Veriseti ve Sonuçlar
Projede kullanılan verisetindeki sınıflar:
- **Cancer**: Kanserli görüntüler
- **Non_Cancer**: Kanser olmayan görüntüler

### Değerlendirme Metrikleri
- **Accuracy**: Doğruluk oranı
- **Precision**: Kesinlik
- **Recall**: Duyarlılık
- **F1-Score**: Harmonik ortalama

#### CNN Sonuçları:
- Accuracy: 0.78
- Precision: 0.54
- Recall: 0.87
- F1-Score: 0.67

#### Random Forest Sonuçları:
- Accuracy: 0.86
- Precision: 0.89
- Recall: 0.53
- F1-Score: 0.67


## Karşılaştırma Sonucu
- **CNN** modeli daha derin özellikler öğrenme konusunda başarılıdır, ancak doğruluk oranı **Random Forest**'tan düşüktür.
- **Random Forest** ise basit bir model olmasına rağmen daha yüksek bir doğruluk oranı sunmaktadır.

