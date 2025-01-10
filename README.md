# Using XAI Algorithms for Material Recognition 

This project uses the ChemBERTa model for molecular property predictions based on SMILES strings. The model is fine-tuned using a dataset of SMILES strings and associated molecular properties.


The frontend for the **Material Recognition Model** is built using **Next.js**, a React-based framework that supports server-side rendering and optimized performance. This interface allows users to input **SMILES** strings and receive predictions for molecular properties.

## Prerequisites

- **Node.js** (v14 or later)
- **npm** package manager
## Project Structure

- `pages/`: Contains React components mapped to different routes.
- `public/`: Static files, such as images and icons.
- `components/`: Reusable React components (e.g., input forms, result cards).
- `styles/`: CSS modules and global styles for the frontend.
- `package.json`: Configuration for dependencies and scripts.

## Features of the Frontend

### 1. **Homepage**

- Welcomes users to the application.
- Provides navigation links to the main features.

![Homepage](https://imgur.com/4kNVN4O.png)
*Description*: The homepage provides quick access to the input and history pages.



### 2. **User Dashboard**

- Personalized dashboard for logged-in users.
- Displays a history of previous predictions.
- Allows users to update their profiles.

![Alt text](https://imgur.com/s748hPu.png)  
*Description*: The user dashboard shows a history of SMILES predictions and profile settings.
---

### 3. **Input SMILES Interface**

- A text input where users can enter **SMILES** strings.
- A button to submit the input for property prediction.

![Input SMILES](https://imgur.com/QyhM5tr.png)  
*Description*: This interface enables users to submit their SMILES strings for prediction.

---

### 4. **Prediction Results**

- Displays the predicted molecular properties in a structured format.
- Shows key molecular descriptors and properties.

![Prediction Results](https://imgur.com/0u1czda.png)
![ChatGPT integration](https://imgur.com/fRd4EVQ.png)
*Description*: The results page shows the predicted properties of the molecule.

---

### 5. **Data Visualization**

- Provides graphical representations of molecular data.
- Allows users to visualize trends and compare predictions.

![Alt text](https://imgur.com/HLPgUax.png)  
![Alt text](https://imgur.com/jn8o0Hp.png)  
![Alt text](https://imgur.com/U4DCg42.png)
*Description*: A graphical summary of molecular properties.

---


## Installation and Setup

### 1. Clone the Repository

If you haven't cloned the repository yet:

``bash
git clone https://github.com/Yassa122/Material-Recognition-Using-XAI-Bachelor-Thesis.git
cd Material-Recognition-Using-XAI-Bachelor-Thesis/frontend
``

### 2. Install Dependencies

Install the required **Node.js** dependencies:

``bash
npm install
``

This will install all packages specified in `package.json`, including **Next.js**, **React**, and other frontend libraries.

### 3. Start the Development Server

To run the frontend application locally:

```bash
npm run dev
```

The application will be available at [http://localhost:3000](http://localhost:3000).

### 4. Build for Production (Optional)

To create an optimized production build:

```bash
npm run build
```

To start the production server:

```bash
npm start
```


---

## Key Technologies

- **Next.js**: A React framework for building fast, server-rendered applications.
- **React**: JavaScript library for building interactive user interfaces.
- **Axios**: Used for making HTTP requests to the backend.
- **Chart.js**: For data visualization within the frontend.

## Development Notes

- Ensure the backend API is running so the frontend can retrieve predictions.
- Use `.env.local` to store sensitive environment variables such as API URLs.

## Customization

You can customize the styles in the `styles/` directory by modifying the CSS modules or global styles.

## Deployment

To deploy the frontend application, follow these steps:

1. Create a production build:

    ```bash
    npm run build
    ```


2. Deploy to your hosting platform of choice:
   - **Vercel**: Recommended platform for **Next.js**.
   - **Netlify**: Another option for serverless deployment.
   - **Docker**: Use `Dockerfile` for containerized deployment.

## Additional Resources

## Prerequisites

- Python 3.8 or later
- `pip` package manager


## Installation 

1. **Clone the Repository**: First, clone the repository to your local machine.

    ```bash
    git clone https://github.com/Yassa122/Material-Recognition-Using-XAI-Bachelor-Thesis.git
    cd Material-Recognition-Using-XAI-Bachelor-Thesis/Model/src/Transformer_model
    ```

2. **Set up a Python virtual environment**: Create and activate a virtual environment to install dependencies.

    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows use `venv\Scripts\activate`
    ```

3. **Install Dependencies**: Install the necessary Python packages. These are listed in `requirements.txt`.

    ```bash
    pip install -r requirements.txt
    ```

   If the `requirements.txt` file is not available, manually install the necessary packages:

    ```bash
    pip install pandas torch transformers scikit-learn tqdm
    ```

## Files and Folder Structure

- `Model/src/Transformer_model/`: Contains scripts and datasets for fine-tuning and prediction.
- `fine_tuned_chemberta/`: Folder where the fine-tuned model will be saved.
- `SMILES_Big_Data_Set.csv`: Dataset for fine-tuning the model.
- `test.csv`: Dataset for making predictions.

## Running the Model

### 1. Prepare the Datasets

Ensure you have the following CSV files in the specified directory:

- `SMILES_Big_Data_Set.csv`: This should contain SMILES strings and molecular properties for training.
- `test.csv`: This should contain SMILES strings without properties for prediction.

### 2. Fine-Tune the Model

Run the following script to fine-tune the ChemBERTa model using the SMILES strings in `SMILES_Big_Data_Set.csv`.

``markdown
# Frontend for Material Recognition Model with ChemBERTa


- **Next.js Documentation**: [https://nextjs.org/docs](https://nextjs.org/docs)
- **React Documentation**: [https://reactjs.org/docs](https://reactjs.org/docs)

