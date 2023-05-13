import os
import sys
from datetime import datetime, timedelta
import atexit
import shutil
import yaml
from typing import Optional

class Documenter:
    """
    Class that makes network runs self-documenting. All output data including the saved
    model, log file, parameter file and plots are saved into an output folder.
    """

    @staticmethod
    def from_saved_run(run_name: str, read_only: bool = False) -> tuple["Documenter", dict]:
        """
        Create a documenter from an existing output folder. Return the documenter and the
        parameter dictionary.

        Args:
            run_name: Used to name the output folder of the run
            read_only: If set, do not overwrite log file

        Returns:
            doc: Created documenter object
            params: Dictionary with the loaded parameters
        """
        with open(run_name) as f:
            params = yaml.load(f, Loader=yaml.FullLoader)
        
        run_name = params.get('run_name')
        doc = Documenter(run_name, existing_run=run_name, read_only=read_only)
        return doc, params

    @staticmethod
    def from_param_file(param_file: str) -> tuple["Documenter", dict]:
        """
        Create a documenter with the run name from param_file and copy the parameter file
        into the output folder. Return the documenter and the parameter dictionary.

        Args:
            param_file: Path to the YAML parameter file

        Returns:
            doc: Created documenter object
            params: Dictionary with the loaded parameters
        """
        with open(param_file) as f:
            params = yaml.load(f, Loader=yaml.FullLoader)
        doc = Documenter(params["run_name"])
        shutil.copy(param_file, doc.add_file("params.yaml", False))
        return doc, params

    def __init__(
        self,
        run_name: str,
        existing_run: Optional[str] = None,
        read_only: bool = False
    ):
        """
        If existing_run is None, a new output folder named as run_name prefixed by date
        and time is created. stdout and stderr are redirected into a log file. The method
        close is registered to be automatically called when the program exits.

        Args:
            run_name: Used to name the output folder of the run
            existing_run: Optionally, the name of an existing run, that should be continued
            read_only: If set, do not overwrite log file
        """
        self.run_name = run_name
        script_dir = os.path.dirname(os.path.abspath(__file__))
        if existing_run is None:
            now = datetime.now()
            while True:
                full_run_name = now.strftime("%Y%m%d_%H%M%S") + "_" + run_name
                self.basedir = os.path.join(script_dir, "..", "output", full_run_name)
                try:
                    os.mkdir(self.basedir)
                    break
                except FileExistsError:
                    now += timedelta(seconds=1)
        else:
            self.basedir = os.path.join(script_dir, "..", "output", existing_run)

        if not read_only:
            self.tee = Tee(self.add_file("log.txt", False))
            atexit.register(self.close)

    def add_file(self, name: str, add_run_name: bool = False) -> str:
        """
        Returns the path in the output folder for a file with the given name. If a file with
        the same name already exists in the output folder, it is moved to a subfolder 'old'.

        Args:
            name: File name
            add_run_name: If True, append run name to file name

        Returns:
            Path to the file in the output folder
        """
        new_file = self.get_file(name, add_run_name)
        old_dir = os.path.join(self.basedir, "old")
        if os.path.exists(new_file):
            os.makedirs(old_dir, exist_ok=True)
            shutil.move(new_file, os.path.join(old_dir, os.path.basename(new_file)))
        return new_file

    def get_file(self, name: str, add_run_name: bool = False) -> str:
        """
        Returns the path in the output folder for a file with the given name. If
        add_run_name is True, the run name is appended to the file name.

        Args:
            name: File name
            add_run_name: If True, append run name to file name

        Returns:
            Path to the file in the output folder
        """
        if add_run_name:
            name_base, name_ext = os.path.splitext(name)
            name = f"{name_base}_{self.run_name}{name_ext}"
        return os.path.join(self.basedir,name)

    def close(self):
        """
        Ends redirection of stdout and changes the file permissions of the output folder
        such that other people on the cluster can access the files.
        """
        self.tee.close()
        os.system("chmod -R 755 " + self.basedir)

class Tee:
    """
    Class to replace stdout and stderr. It redirects all printed data to std_out as well
    as a log file.
    """

    def __init__(self, log_file: str):
        """
        Creates log file and redirects stdout and stderr.

        Args:
            log_file: Path to the log file
        """
        self.log_file = open(log_file, "w")
        self.stdout = sys.stdout
        self.stderr = sys.stderr
        sys.stdout = self
        sys.stderr = self

    def close(self):
        """
        Closes log file and restores stdout and stderr.
        """
        sys.stdout = self.stdout
        sys.stderr = self.stderr
        self.log_file.close()

    def write(self, data):
        """
        Writes data to stdout and the log file.

        Args:
            data: Data to be written to stdout
        """
        self.log_file.write(data)
        self.stdout.write(data)

    def flush(self):
        """
        Flushes buffered data to the file.
        """
        self.log_file.flush()
