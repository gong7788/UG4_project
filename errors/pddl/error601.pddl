(define (problem block-problem)
	(:domain blocksworld)
	(:objects b0 b1 b2 b3 b4 b5 b6 b7 b8 b9 t0)
	(:init 
		(arm-empty )
		(on-table b0)
		(clear b0)
		(clear b1)
		(on-table b2)
		(clear b2)
		(on-table b3)
		(clear b3)
		(in-tower t0)
		(on b9 t0)
		(in-tower b9)
		(on b8 b9)
		(in-tower b8)
		(on b7 b8)
		(in-tower b7)
		(on b5 b7)
		(in-tower b5)
		(on b4 b5)
		(in-tower b4)
		(on b6 b4)
		(in-tower b6)
		(on b1 b6)
		(in-tower b1)
		(blue b0)
		(green b1)
		(maroon b1)
		(red b3)
		(blue b4)
		(green b5)
		(maroon b5)
		(red b6)
		(blue b6)
		(red b1)
	)
	(:goal (and (forall (?x) (in-tower ?x)) (forall (?x) (or (not (red ?x)) (exists (?y) (and (blue ?y) (on ?x ?y))))) (forall (?y) (or (not (blue ?y)) (exists (?x) (and (red ?x) (on ?x ?y))))) (forall (?x) (or (not (green ?x)) (exists (?y) (and (maroon ?y) (on ?x ?y))))) (forall (?y) (or (not (maroon ?y)) (exists (?x) (and (green ?x) (on ?x ?y))))) (and (not (on b6 b7)) (not (on b6 b5)) (not (on b3 b6)) (not (on b2 b6)) (not (on b3 b1)))))
)