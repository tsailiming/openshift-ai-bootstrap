#!/bin/sh

echo "This will sync /mnt/models/* into s3://models"
aws s3 sync /mnt/models s3://models
