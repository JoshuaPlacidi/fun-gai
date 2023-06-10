import React, { Component } from "react";

class SpeciesList extends Component {
  constructor(props) {
    super(props);
    this.state = {
      species: null
    };

   // if we are using arrow function binding is not required
   //  this.onImageChange = this.onImageChange.bind(this);
  }

  render() {
    return (
        <div>
            <div>Species 1</div>
            <div>Species 2</div>
            <div>Species 3</div>
            <div>Species 4</div>
            <div>Species 5</div>
      </div>
    );
  }
}
export default SpeciesList;