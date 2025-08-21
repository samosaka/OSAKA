app.controller('MainCtrl', function($scope, $timeout) {
  // Shared logic can go here
});




// app.controller('MainCtrl', function ($scope, $timeout) {
//     let vm = this;

//     $('#normal-contained').dxButton({
//         stylingMode: 'contained',
//         text: 'Contained',
//         type: 'normal',
//         width: 120,
//         onClick() {
//             DevExpress.ui.notify('The Contained button was clicked');
//         },
//     });
//     // Use $timeout to ensure DOM is ready

//     $("#gridContainer").dxDataGrid({
//         dataSource: [
//             { ID: 1, Name: "Alice", Age: 28 },
//             { ID: 2, Name: "Bob", Age: 34 },
//             { ID: 3, Name: "Charlie", Age: 45 }
//         ],
//         columns: ["ID", "Name", "Age"],
//         showBorders: true
//     });
// });