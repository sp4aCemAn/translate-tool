import 'package:bloc/bloc.dart';
import 'package:flutter/material.dart';
import 'navigation_bloc.dart';


class NavigationPage extends StatelessWidget {
  const NavigationPage({super.key});

  @override 
    Widget build(BuildContext context) {
      return Scaffold(
          appBar: AppBar(
            title: Text("Translator App"),

            ),
            drawer: Drawer( 
              child: ListView(
              padding: EdgeInsets.zero,
                

              )
            ),
          );

    }

}
