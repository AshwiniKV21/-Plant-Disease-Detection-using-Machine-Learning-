from django.db import models

class LeafScan(models.Model):
    image          = models.ImageField(upload_to='leaf_images/')
    plant_name     = models.CharField(max_length=100, blank=True, default="Unknown")
    created_at     = models.DateTimeField(auto_now_add=True)

    # Prediction Output
    predicted_class = models.CharField(max_length=100)    # e.g. "Tomato___Late_Blight"
    is_healthy       = models.BooleanField(default=False)
    confidence        = models.FloatField()                 # confidence %
    severity           = models.CharField(max_length=20, default="None")  # Low/Moderate/High/None

    def _str_(self):
        return f"{self.predicted_class} ({self.confidence}%) — {self.created_at.strftime('%Y-%m-%d')}"

    class Meta:
        ordering = ['-created_at'] 