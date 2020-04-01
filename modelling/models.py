from django.db import models


class Dataset(models.Model):
    openml_idx = models.IntegerField()
    title = models.CharField(max_length=128)
    categorical_names = models.TextField()
    attribute_names = models.TextField()
    contrast_names = models.TextField()
    columns_to_drop = models.TextField()

    def _str_(self):
        return self.title
