from django.db import models

class Texts(models.Model):
    id = models.AutoField(primary_key=True)
    text = models.TextField()
    textType = models.IntegerField()

    def __str__(self):
        return f'Text ID: {self.id}, Type: {self.textType}'


class ProcessedTexts(models.Model):
    id = models.AutoField(primary_key=True)
    original_text = models.TextField()
    processed_text = models.TextField()
    textType = models.IntegerField()

    def __str__(self):
        return f'Processed Text ID: {self.id}, Type: {self.textType}'


class UserProcessedTexts(models.Model):
    id = models.AutoField(primary_key=True)
    processed_text = models.TextField()
    category=models.TextField(blank=True)
    def __str__(self):
        return f'User Processed Text ID: {self.id}'

