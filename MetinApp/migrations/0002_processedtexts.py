# Generated by Django 5.1.2 on 2024-10-30 10:12

from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('MetinApp', '0001_initial'),
    ]

    operations = [
        migrations.CreateModel(
            name='ProcessedTexts',
            fields=[
                ('id', models.AutoField(primary_key=True, serialize=False)),
                ('original_text', models.TextField()),
                ('processed_text', models.TextField()),
                ('textType', models.IntegerField()),
            ],
        ),
    ]
