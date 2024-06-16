const imageUpload = document.getElementById("imageUpload");
const imagePreview = document.getElementById("imagePreview");
const resultSection = document.getElementById("result");

imageUpload.addEventListener("change", async function () {
  resultSection.innerHTML = `<p>Processing...</p>`;
  const file = this.files[0];

  //Previewing image
  if (file) {
    const reader = new FileReader();

    reader.addEventListener("load", function () {
      imagePreview.src = reader.result;
    });

    reader.readAsDataURL(file);
  } else {
    imagePreview.src = "";
  }

  // Calling api endpoint
  const formData = new FormData();
  formData.append("image", imageUpload.files[0]);

  try {
    const response = await fetch("http://127.0.0.1:5000", {
      method: "POST",
      body: formData,
    });

    if (!response.ok) {
      throw new Error("Failed to upload image");
    }

    const res = await response.json();
    resultSection.innerHTML = `<p>Result: ${res[0].result}</p>`;
    console.log("API Response:", res);

    // Handle API response as needed
  } catch (error) {
    console.error("Error uploading image:", error);
  }
});
