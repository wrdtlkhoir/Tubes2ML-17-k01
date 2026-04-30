import os
from cnn.train import ModelTrainer


def main(train_dir, val_dir, test_dir, output_dir='./trained_models', epochs=20):
    trainer = ModelTrainer(train_dir, val_dir, test_dir, output_dir)
    
    print("Loading dataset...")
    train_ds, val_ds, test_ds = trainer.load_data(img_size=(224, 224), batch_size=32)
    
    print("Starting training...")
    trainer.run_all_training(train_ds, val_ds, test_ds)
    
    print("\nTraining complete!")


if __name__ == '__main__':
    train_dir = './data/train'
    val_dir = './data/val'
    test_dir = './data/test'
    
    main(train_dir, val_dir, test_dir)
