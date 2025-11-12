import { app } from "../../../scripts/app.js";
import { $el } from "../../../scripts/ui.js";

function createComplexWidget(node, inputName, inputData, app) {
    const widget = {
        type: inputData[0],  // the type of the input data
        name: inputName,     // the name of the widget
        size: [400, 30],     // default size
        inputEl: $el("div", { class: "complex-widget" }),  // create a div element
        values: { bool1: false, float1: 0.0, float2: 0.0, bool2: false, bool3: false },

        // Method to draw the widget
        draw(ctx, node, width, y) {
            this.inputEl.style.left = `${node.pos[0]}px`;
            this.inputEl.style.top = `${node.pos[1] + y}px`;
            this.inputEl.style.width = `${width}px`;
            this.inputEl.innerHTML = "";

            // Create input fields for each value
            const bool1 = $el("input", { type: "checkbox", checked: this.values.bool1 });
            bool1.onchange = () => {
                this.values.bool1 = bool1.checked;
                node.onWidgetChanged(this);
            };
            this.inputEl.appendChild(bool1);

            const float1 = $el("input", { type: "number", value: this.values.float1 });
            float1.onchange = () => {
                this.values.float1 = parseFloat(float1.value);
                node.onWidgetChanged(this);
            };
            this.inputEl.appendChild(float1);

            const float2 = $el("input", { type: "number", value: this.values.float2 });
            float2.onchange = () => {
                this.values.float2 = parseFloat(float2.value);
                node.onWidgetChanged(this);
            };
            this.inputEl.appendChild(float2);

            const bool2 = $el("input", { type: "checkbox", checked: this.values.bool2 });
            bool2.onchange = () => {
                this.values.bool2 = bool2.checked;
                node.onWidgetChanged(this);
            };
            this.inputEl.appendChild(bool2);

            const bool3 = $el("input", { type: "checkbox", checked: this.values.bool3 });
            bool3.onchange = () => {
                this.values.bool3 = bool3.checked;
                node.onWidgetChanged(this);
            };
            this.inputEl.appendChild(bool3);
        },

        // Method to compute the size of the widget
        computeSize() {
            return [this.size[0], this.size[1]];
        },

        // Method to serialize the widget's value
        async serializeValue(nodeId, widgetIndex) {
            return this.values;
        }
    };

    // Add the widget to the document body
    document.body.appendChild(widget.inputEl);
    node.addCustomWidget(widget);  // Add the widget to the node

    // Ensure the widget is removed from the document when the node is removed
    node.onRemoved = () => { widget.inputEl.remove(); };
    node.serialize_widgets = false;

    return widget;
}

// Register the custom widget with ComfyUI
app.registerExtension({
    name: "my.unique.extension.name",
    async beforeRegisterNodeDef(nodeType, nodeData, app) {
        if (nodeType.comfyClass === "ComplexValueNode") {
            const orig_nodeCreated = nodeType.prototype.onNodeCreated;
            nodeType.prototype.onNodeCreated = function () {
                orig_nodeCreated?.apply(this, arguments);
                const widget = createComplexWidget(this, "complex_value", ["COMPLEX"], app);
                this.addCustomWidget(widget);
            };
        }
    }
});