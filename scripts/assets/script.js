//alert('If you see this alert, then your custom JavaScript script has run!');
// window.onbeforeunload = function(event) {
//     event.returnValue = "Êtes-vous sûr de vouloir quitter? Les changements non sauvegardés seront perdus.";
//     return "Êtes-vous sûr de vouloir quitter? Les changements non sauvegardés seront perdus.";
// };
document.body.onbeforeunload = function(event) {
    console.log("This message is printed in the browser console");
    //var sessionData = localStorage.getItem('info-current-file-store'); // Récupérer les données de session
    try {
        // Code à essayer
        // Par exemple:
        //var sessionData = document.getItem('info-current-file-store'); // Récupérer les données de session
        let sessionData = sessionStorage.getItem('info-current-file-store');
        console.log("COucou toi");
        console.log(sessionData);
    } catch (error) {
        // Code à exécuter en cas d'erreur
        console.error("Une erreur s'est produite :", error);
    }
    
    //console.log(JSON.stringify({sessionData}));
    let sessionData = sessionStorage.getItem('info-current-file-store');
    fetch('/close-session', {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json'
        },
        body: sessionData
        //body: sessionData
    }).then(response => response.json())
      .then(data => console.log(data))
      .catch((error) => console.error('Error:', error));
    return "Write something clever here..."
};

// document.body.onunload = function(event) {
//     console.log("Onlunload");
//     fetch('/close-session', {
//         method: 'POST',
//         headers: {
//             'Content-Type': 'application/json'
//         },
//         body: JSON.stringify({'message': 'Window closed'})
//     }).then(response => response.json())
//       .then(data => console.log(data))
//       .catch((error) => console.error('Error:', error));
//     return "Write something clever here..."
// };

// console.log("This message is for test");

// window.addEventListener('beforeunload', (event) => {
//     console.log("COucou toi ")
//     event.preventDefault();
//     event.returnValue = '';
//   });

// function goodBye()
//   {
//       alert("Goodbye, " + promptName() + "!");
//   }

//   window.onunload = goodBye();