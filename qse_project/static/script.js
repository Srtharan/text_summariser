 <script>
        document.getElementById('summarize-btn').addEventListener('click', function() {
            var inputText = document.getElementById('input-text').value;
            alert("Summarising...");

            // Make AJAX request to Flask backend to get summary
            fetch('/summarize', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json'
                },
                body: JSON.stringify({ "text": inputText })
            }).then(response => response.json()).then(data => {
                console.log(data);
                document.getElementById('summary-text').textContent = data.summary;
                document.getElementById('output-section').style.display = 'block';
            })
            .catch(error => console.error('Error:', error));
        });
<!---->
    document.getElementById('summarize-pdf-btn').addEventListener('click', function() {
    var fileInput = document.getElementById('file-input').files[0];
    alert(fileInput.name);
    console.log(fileInput);
function extractText(pdfUrl) {
	var pdf = pdfjsLib.getDocument(pdfUrl);
	return pdf.promise.then(function (pdf) {
		var totalPageCount = pdf.numPages;
		var countPromises = [];
		for (
			var currentPage = 1;
			currentPage <= totalPageCount;
			currentPage++
		) {
			var page = pdf.getPage(currentPage);
			countPromises.push(
				page.then(function (page) {
					var textContent = page.getTextContent();
					return textContent.then(function (text) {
						return text.items
							.map(function (s) {
								return s.str;
							})
							.join('');
					});
				}),
			);
		}

		return Promise.all(countPromises).then(function (texts) {
			return texts.join('');
		});
	});
}
    // Use FormData to send the file
    var formData = new FormData();
    formData.append('file', fileInput);
    console.log(formData);

    // Make AJAX request to Flask backend to get summary
    fetch('/summarizepdf', {
        method: 'POST',
        headers: {
                    'Content-Type': 'application/json'
                },
                body: JSON.stringify({ "text": formData })
    }).then(response => response.json()).then(data => {
        console.log(data);
        document.getElementById('summary-text').textContent = data.summary;
        document.getElementById('output-section').style.display = 'block';
    })
    .catch(error => console.error('Error:', error));
});

<!---->
        document.getElementById('ques-btn').addEventListener('click', function() {
            var ques = document.getElementById('input-ques').value;
            alert("Your Question is ", ques);
            // Make AJAX request to Flask backend to get summary
            fetch('/answer', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json'
                },
                body: JSON.stringify({ "text": ques })
            }).then(response => response.json()).then(data => {
                alert(data);
                document.getElementById('ans').textContent = data.answer;
                document.getElementById('bot-ans').style.display = 'block';
            })
            .catch(error => console.error('Error:', error));
        });