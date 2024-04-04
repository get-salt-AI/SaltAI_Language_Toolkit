import os
import time
import traceback
import importlib


class ModuleLoader:
    def __init__(self, package):
        self.EXTENSION_WEB_DIRS = {}
        self.NODE_CLASS_MAPPINGS = {}
        self.NODE_DISPLAY_NAME_MAPPINGS = {}
        self.module_timings = {}
        self.package = package

    def import_module(self, module_name, package_name):
        success = True
        error = None
        module = None
        try:
            module = importlib.import_module(module_name, package=package_name)
        except Exception as e:
            error = e
            success = False
            traceback.print_exc()
        return module, success, error

    def update_mappings(self, module, module_name, module_dir):

        # Whenever this is a thing...
        if hasattr(module, "WEB_DIRECTORY") and getattr(module, "WEB_DIRECTORY") is not None:
            web_dir = os.path.abspath(os.path.join(module_dir, getattr(module, "WEB_DIRECTORY")))
            if os.path.isdir(web_dir):
                self.EXTENSION_WEB_DIRS[module_name] = web_dir

        if hasattr(module, 'NODE_CLASS_MAPPINGS'):
            self.NODE_CLASS_MAPPINGS.update(module.NODE_CLASS_MAPPINGS)
        elif hasattr(module, '__all__'):
            for attr_name in module.__all__:
                attr_value = getattr(module, attr_name, None)
                if attr_value is not None and hasattr(attr_value, 'NODE_CLASS_MAPPINGS'):
                    self.NODE_CLASS_MAPPINGS.update(attr_value.NODE_CLASS_MAPPINGS)

        if hasattr(module, 'NODE_DISPLAY_NAME_MAPPINGS'):
            self.NODE_DISPLAY_NAME_MAPPINGS.update(module.NODE_DISPLAY_NAME_MAPPINGS)
        elif hasattr(module, '__all__'):
            for attr_name in module.__all__:
                attr_value = getattr(module, attr_name, None)
                if attr_value is not None and hasattr(attr_value, 'NODE_DISPLAY_NAME_MAPPINGS'):
                    self.NODE_DISPLAY_NAME_MAPPINGS.update(attr_value.NODE_DISPLAY_NAME_MAPPINGS)

    def load_modules(self, NODES_DIR):
        for filename in os.listdir(NODES_DIR):
            if filename.startswith('__') or filename.endswith(".disabled"):
                continue

            filepath = os.path.join(NODES_DIR, filename)

            if os.path.isdir(filepath):
                if os.path.isfile(os.path.join(filepath, '__init__.py')):
                    start_time = time.time()
                    module_name = f".nodes.{filename}"
                    module, success, error = self.import_module(module_name, self.package)
                    end_time = time.time()
                    timing = end_time - start_time
                    if success:
                        self.update_mappings(module, module_name, filepath)
                    self.module_timings[module.__file__ if success else filename] = (timing, success, error)
                else:
                    for sub_filename in os.listdir(filepath):
                        if sub_filename.endswith('.py') and sub_filename != '__init__.py':
                            start_time = time.time()
                            module_name = f".nodes.{filename}.{sub_filename[:-3]}"
                            module, success, error = self.import_module(module_name, self.package)
                            end_time = time.time()
                            timing = end_time - start_time
                            if success:
                                self.update_mappings(module, module_name, filepath)
                            self.module_timings[module.__file__ if success else os.path.join(filename, sub_filename)] = (timing, success, error)
            elif filename.endswith('.py'):
                start_time = time.time()
                module_name = f".nodes.{filename[:-3]}"
                module, success, error = self.import_module(module_name, self.package)
                end_time = time.time()
                timing = end_time - start_time
                if success:
                    self.update_mappings(module, module_name, filepath)
                self.module_timings[module.__file__ if success else filename] = (timing, success, error)
    
    def report(self, NAME):
        print(f"\33[1mImport times for {NAME} Node Modules:\33[0m")
        for module, (timing, success, error) in self.module_timings.items():
            print(f"   {timing:.1f} seconds{('' if success else ' (IMPORT FAILED)')}: {module}")
            if error:
                print("Error:", error)

        loaded_nodes = []
        for class_name in self.NODE_CLASS_MAPPINGS.keys():
            suffix = f' [{self.NODE_DISPLAY_NAME_MAPPINGS[class_name]}]' if self.NODE_DISPLAY_NAME_MAPPINGS.__contains__(class_name) else ''
            loaded_nodes.append(f"\33[1m{class_name}\33[0m{suffix}")

        print(f"\n\33[1mLoaded Nodes:\33[0m\n" + "\33[93m,\33[0m ".join(loaded_nodes))

