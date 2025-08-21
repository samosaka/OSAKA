// app.js
var app = angular.module('myApp', ['ngRoute']);


app.config(function($routeProvider) {
  $routeProvider
    .when("/", {
      templateUrl: "js/views/home.html",
      controller: "HomeCtrl"
    })
    .when("/history", {
      templateUrl: "js/views/history.html",
      controller: "HistoryCtrl"
    })
    .when("/training", {
      templateUrl: "js/views/training.html",
      controller: "TrainingCtrl"
    })
    .when("/testing", {
      templateUrl: "js/views/testing.html",
      controller: "TestingCtrl"
    })
    .when("/contact", {
      templateUrl: "js/views/contact.html",
      controller: "ContactCtrl"
    })
    .otherwise({ redirectTo: "/" });
});

app.factory('loaderInterceptor', function($q, $rootScope) {
  var requestCount = 0;

  function showLoader() {
    requestCount++;
    $rootScope.isLoading = true;
  }

  function hideLoader() {
    requestCount--;
    if (requestCount <= 0) {
      $rootScope.isLoading = false;
    }
  }

  return {
    request: function(config) {
      showLoader();
      return config;
    },
    response: function(response) {
      hideLoader();
      return response;
    },
    responseError: function(response) {
      hideLoader();
      return $q.reject(response);
    }
  };
});

app.config(function($httpProvider) {
  $httpProvider.interceptors.push('loaderInterceptor');
});

