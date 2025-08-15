# Dog Breed Identification Web App

A simple web application built with Node.js and TensorFlow.js that identifies a dog's breed from an uploaded image.

## Features

  - Image upload functionality.
  - A two-step prediction process:
    1.  Verifies if the uploaded image contains a dog.
    2.  Classifies the breed of the dog.
  - Displays the top breed predictions with their confidence scores.

## Tech Stack

  - **Backend:** Node.js, Express
  - **Machine Learning:** TensorFlow.js (`@tensorflow/tfjs-node`)
  - **File Handling:** Multer
  - **Templating:** EJS

## Getting Started

### Prerequisites

  - [Node.js](https://nodejs.org/) installed on your machine.

### Installation & Setup

1.  Clone the repository or save the `server.js` file in a new directory.
2.  Navigate to the project directory in your terminal.
3.  Install the required npm packages:
    ```bash
    npm install express multer @tensorflow/tfjs-node ejs
    ```
4.  You will also need an `index.ejs` file in a `views` directory for the frontend.

### Running the Application

1.  Run the server from your terminal:
    ```bash
    node server.js
    ```
2.  Open your web browser and go to `http://localhost:3000`.
3.  Upload an image of a dog to see the breed prediction.