package controllers

import javax.inject._
import play.api.mvc._
import models.Task
import repositories.TaskRepository
import scala.concurrent.ExecutionContext

@Singleton
class TaskController @Inject()(cc: ControllerComponents, taskRepo: TaskRepository)(implicit ec: ExecutionContext) extends AbstractController(cc) {

  def create() = Action.async { implicit request =>
    val task = Task(0, "Tugas Baru", "Deskripsi Tugas")
    taskRepo.create(task).map { createdTask =>
      Created(s"Tugas dibuat dengan ID: ${createdTask.id}")
    }
  }

  def list() = Action.async {
    taskRepo.findAll().map { tasks =>
      Ok(views.html.taskList(tasks))
    }
  }

  def update(id: Long) = Action.async { implicit request =>
    val updatedTask = Task(id, "Tugas Diperbarui", "Deskripsi Diperbarui")
    taskRepo.update(updatedTask).map {
      case Some(task) => Ok(s"Tugas dengan ID: ${task.id} diperbarui.")
      case None => NotFound("Tugas tidak ditemukan.")
    }
  }

  def delete(id: Long) = Action.async {
    taskRepo.delete(id).map {
      case Some(_) => Ok(s"Tugas dengan ID: $id dihapus.")
      case None => NotFound("Tugas tidak ditemukan.")
    }
  }
}