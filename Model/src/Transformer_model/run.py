# run.py

from app import create_app

app = create_app()

if __name__ == "__main__":
    import multiprocessing

    multiprocessing.freeze_support()
    app.run(debug=True)
